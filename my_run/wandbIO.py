from ray import tune, train
from ray.tune.logger import DEFAULT_LOGGERS
from ray.air.integrations.wandb import WandbLoggerCallback
import wandb

def custom_trial_name_creator(trial):
    name = "my_trial"
    for key, value in trial.config.items():
        name += f"_{key}_{value:.3f}"
    return name

def train_fc(config):
    for i in range(10):
        train.report({"mean_accuracy":(i + config['alpha']) / 10})
        wandb.log({"mean_accuracy":(i + config['alpha']) / 10})

search_space = {
    'alpha': tune.grid_search([0.1, 0.2, 0.3]),
    'beta': tune.uniform(0.5, 1.0)
}

analysis = tune.run(
    train_fc,
    config=search_space,
    callbacks=[WandbLoggerCallback(
        project="ray_wandb",
        log_config=True
    )],
    storage_path=r"file:///C:/Users/W/Desktop/Time-Series-Library/logs/HAR-B",
    trial_name_creator=custom_trial_name_creator,
    trial_dirname_creator=custom_trial_name_creator,
)

best_trial = analysis.get_best_trial("mean_accuracy", "max", "last")