from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TCAttention(nn.Module):
    def __init__(self, input_shape):
        super(TCAttention, self).__init__()
        channel_enhance_dict = {6: 14, 9: 37}
        batch_size, channel, seq_len = input_shape

        # self.ForwardConv = nn.Sequential(
        #     nn.Conv1d(channel_enhance_dict[channel], channel, kernel_size=1),
        # )

        self.CAM = nn.Sequential(
            nn.AvgPool2d((1, seq_len)),
            nn.Conv1d(channel, channel // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channel // 2, channel, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        bs = x.shape[0]
        # x = self.ForwardConv(x)
        delta = self.CAM(x)
        x = x + delta * x
        return x


class tAPE(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        x = x + self.pe
        return self.dropout(x)


class eRPE(nn.Module): # Equation 14 page 12
    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)
        self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.seq_len - 1), num_heads))

        coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)))
        coords = torch.flatten(torch.stack(coords), 1)

        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        # print("relative_bias_table:",self.relative_bias_table.shape)
        # print("relative_index:",relative_index.shape)
        self.register_buffer("relative_index", relative_index)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)
        attn = torch.matmul(q, k) * self.scale
        attn = nn.functional.softmax(attn, dim=-1) # attn shape (seq_len, seq_len)
        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.num_heads))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=1 * self.seq_len, w=1 * self.seq_len)
        attn = attn + relative_bias
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2) # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.reshape(batch_size, seq_len, -1) # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = self.to_out(out) # out.shape == (batch_size, seq_len, d_model)
        return out


class attenModule(nn.Module):
    def __init__(self, inputData, embed_dim, num_heads, dropout):
        super().__init__()
        batch_size, seq_len, channel_num = inputData

        self.attention_layer = eRPE(emb_size=embed_dim, num_heads=num_heads, seq_len=seq_len, dropout=dropout)

        self.FeedForward = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.Dropout(dropout))

        self.LayerNorm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(embed_dim, eps=1e-5)

    def forward(self, x):
        att = x + self.attention_layer(x)
        att = self.LayerNorm1(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        return out


class eREPEncoder(nn.Module):
    def __init__(self, inputData, embed_dim, class_num, layer_num=3, num_heads=4, encoder=True):
        super().__init__()
        batch_size, seq_len, channel_num = inputData
        self.encoder = encoder
        dropout = 0.1
        # Embedding Layer -----------------------------------------------------------
        self.embed_dim = embed_dim
        self.embed_layer = nn.Sequential(nn.Conv2d(1, embed_dim * 4, kernel_size=[1, num_heads], padding='same'),
                                         nn.BatchNorm2d(embed_dim * 4),
                                         nn.GELU())

        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=[channel_num, 1], padding='valid'),
            nn.BatchNorm2d(embed_dim),
            nn.GELU())

        self.Fix_Position = tAPE(embed_dim, dropout=dropout, max_len=seq_len)

        self.Channel_attention = TCAttention((batch_size, channel_num, seq_len))

        self.attention_layers = nn.ModuleList([attenModule(inputData, embed_dim, num_heads, dropout) for _ in range(layer_num)])

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.LayerNorm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.fc_out = nn.Linear(embed_dim, class_num)

    def forward(self, x):
        bs = x.shape[0]
        x = self.Channel_attention(x)
        x = x.unsqueeze(1)
        x = self.embed_layer(x)
        x = self.embed_layer2(x).squeeze(2)
        x = x.permute(0, 2, 1)
        x = self.Fix_Position(x)

        for layer in self.attention_layers:
            x = layer(x)  # 对每个注意力层调用 forward 函数

        out = x.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        if self.encoder:
            return out
        out = self.fc_out(out)
        return out


class TFeREPModel(nn.Module):
    def __init__(self, inputData, num_class, embed_dim=128, eval_type="encoder"):
        super(TFeREPModel, self).__init__()
        # batch_size, seq_len, channel_num = inputData

        self.training = True
        self.loss_scale = nn.Parameter(torch.tensor([-0.5]*4))

        self.embed_dim = embed_dim
        self.eval_type = eval_type
        self.proj_dim = 64

        self.time_encoder = eREPEncoder(inputData, self.embed_dim,10,3,True)
        self.spec_encoder = eREPEncoder(inputData, self.embed_dim,10,3,True)

        self.fc_out = nn.Linear(self.proj_dim*2, num_class)
        self.fc_out2 = nn.Linear(self.proj_dim*4, num_class)

        self.fc_out_ce1 = nn.Linear(self.proj_dim*2,num_class)
        self.fc_out_ce2 = nn.Linear(self.proj_dim*2,num_class)

        self.projector_t = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.proj_dim),
        )
        self.projector_f = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.proj_dim),
        )


    def _DataTransform_FD(self,sample):
        """Weak and strong augmentations in Frequency domain """
        aug_1 = self._remove_frequency(sample, pertub_ratio=0.4)
        aug_2 = self._add_frequency(sample, pertub_ratio=0.4)
        aug_F = aug_1 + aug_2
        return aug_F


    def _remove_frequency(self, x, pertub_ratio=0.0):
        mask = torch.FloatTensor(x.shape).uniform_() > pertub_ratio  # maskout_ratio are False
        mask = mask.to(x.device)
        return x * mask


    def _add_frequency(self, x, pertub_ratio=0.0):

        mask = torch.FloatTensor(x.shape).uniform_() > (
                    1 - pertub_ratio)  # only pertub_ratio of all values are True
        mask = mask.to(x.device)
        max_amplitude = x.max()
        random_am = torch.rand(mask.shape).to(x.device) * (max_amplitude * 0.1)
        pertub_matrix = mask * random_am
        return x + pertub_matrix


    def _DataTransform_TD(self, x):
        dropout_rate = 0.07
        x = x.unsqueeze(1)
        x = torch.cat([x, F.dropout(x, p=dropout_rate, training=True)], dim=1)  # 直接在原地操作上应用dropout
        x = torch.cat([x[:, 0, :, :], x[:, 1, :, :]], dim=0)
        return x

    # (batch_size, self.n_vars, self.sequenceLen)
    def forward(self,x):
        if self.training:
            x_time = self._DataTransform_TD(x)
            x_spec = torch.fft.fft(x_time).abs()
            # x_spec = torch.fft.fft(x).abs()
            # x_spec = torch.cat([x_spec,self._DataTransform_FD(x_spec)], dim=0)

        else:
            x_time = x
            x_spec = torch.fft.fft(x).abs()

        time_feature = self.time_encoder(x_time)
        spec_feature = self.spec_encoder(x_spec)

        if self.eval_type == "encoder":
            time_feature_z = self.projector_t(time_feature)
            spec_feature_z = self.projector_f(spec_feature)
            return time_feature, time_feature_z, spec_feature, spec_feature_z
        elif self.eval_type == "ce":
            # fusion_feature = self.fusion_model(time_feature, spec_feature)
            out = self.fc_out(F.relu(time_feature))
        elif self.eval_type == "fusion":
            time_feature_z = self.projector_t(time_feature)
            spec_feature_z = self.projector_f(spec_feature)

            # SCL 用
            # out = self.fc_out(F.relu(torch.cat([time_feature_z,spec_feature_z],dim=-1))) # for loss_T + loss_F + loss_cross
            out = self.fc_out2(F.relu(torch.cat([time_feature,spec_feature],dim=-1))) # for loss_T + loss_F
            # out = self.fc_out(F.relu(time_feature)) # for loss_T
            # out = self.fc_out(F.relu(spec_feature)) # for loss_F

            # CE 用
            # out_t = self.fc_out_ce1(F.relu(time_feature))
            # out_f = self.fc_out_ce2(F.relu(spec_feature))

            # out = (out_t+out_f)/2 # for sum
            # out = torch.max(out_t,out_f) # for max
            # out = self.fc_out2(F.relu(torch.cat([time_feature,spec_feature],dim=-1))) # for concat

            return time_feature,time_feature_z,spec_feature,spec_feature_z,out
