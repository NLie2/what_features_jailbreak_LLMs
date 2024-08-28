from train import train
import wandb
wandb.login()

sweep_config  = {
    "method": "random",
    "metric": {
        "goal": "maximize",
        "name": "highest_val_accuracy"
    },
    "parameters": {
        "layer": {
            # "values": [1,2,3]
            "value": 17
        },
        "dataset": {
            # "values": ["wei", "wildjailbreak", "both"]
            "value": "wei"
        },
        "lr": {
            # "values": [0.1, 0.01, 0.001, .0001, .000005]
            "value": 0.01
        },
        "model": {
            "values": ["linear_probe", "mlp_hidden_8", "mlp_hidden_16", "mlp_hidden_64"]
        }
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project="project-alpha")
wandb.agent(sweep_id, train)