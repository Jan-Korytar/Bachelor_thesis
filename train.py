import torch
import yaml

from utilities.train import train

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('config.yaml', 'r') as file:
    file = yaml.safe_load(file)
    config = file['config_segmentation_model']
    # config = file['config_cropping_model']



if __name__ == '__main__':
    train(config)
