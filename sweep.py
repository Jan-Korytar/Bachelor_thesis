import wandb
import yaml

from utilities.train import train_sweep

# Load the sweep configuration from YAML file
with open('config.yaml', 'r') as file:
    sweep_config = yaml.safe_load(file)['wandb_sweep']


# Function to run the training for each sweep iteration
def sweep_train():
    # Pass the configuration to the training function
    with wandb.init():
        # Now `wandb.config` is populated with the current sweep parameters
        config = wandb.config
        train_sweep(config)

if __name__ == '__main__':

    # Initialize sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project="sweeping")

    # Start the sweep, running as many iterations as you need
    wandb.agent(sweep_id, function=sweep_train)
