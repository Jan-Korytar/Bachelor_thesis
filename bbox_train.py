import numpy as np
import torchvision
torchvision.disable_beta_transforms_warning()

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import BboxDataset
from paths import get_preprocessed_images_paths
from bbox_model import bbox_model
import wandb
import torch.optim as optim
from torchvision.ops import masks_to_boxes, generalized_box_iou_loss, complete_box_iou_loss, distance_box_iou_loss
wandb.login()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_of_all_val_loss = -torch.inf




def main():

    wrong_output_dim = False
    config = {
        'batch_size': 1,
        'lr': 1.781623386838983e-06,
        'base_dim': 60,
        'dropout': 0.3,
        'batch_norm': True,
        'loss_type': 'distance_iou',
        'decay': .728363910889139,
        'normalize_images': False

    }
    with wandb.init(project='sweep_1', config=config):
        config = wandb.config
        batch_size = config.batch_size

        train_images, train_masks, val_images, val_masks, test_images, test_masks = get_preprocessed_images_paths()

        train_dataset = BboxDataset(train_images, train_masks, normalize_images=config.normalize_images)
        val_dataset = BboxDataset(val_images, val_masks, normalize_images=config.normalize_images)

        val_loader = DataLoader(val_dataset, batch_size=64)

        # Define your model
        model = bbox_model(in_channels=3, base_dim=config.base_dim, dropout=config.dropout, batch_norm=config.batch_norm).to(device)
        wandb.watch(model, log_freq=20)

        # Define your optimizer
        optimizer = optim.Adam(model.parameters(), lr=config.lr)


        # Train the model
        torch.cuda.empty_cache()
        best_val_loss = torch.inf
        tolerance = 4

        for epoch in range(num_epochs := 25):
            sampler = torch.utils.data.SubsetRandomSampler(torch.randint(high=len(train_images), size=(400, ), ))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            # Train
            model.train()
            train_loss = 0
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=config.decay, step_size=2)
            bboxes_mean = []
            #if epoch == 3:
            #    for param_group in optimizer.param_groups:
            #        param_group['lr'] = param_group['lr']*5

            for idx, (images, masks) in tqdm(enumerate(train_loader), total=len(train_loader)):
                images = images.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()
                outputs = model(images) * 126
                '''
                a = torch.clamp(outputs[:, 0], max=outputs[:, 2])
                b = torch.clamp(outputs[:, 1], max=outputs[:, 3])
                c = torch.clamp(outputs[:, 2], min=outputs[:, 0])
                d = torch.clamp(outputs[:, 3], min=outputs[:, 1])
                outputs = torch.stack([a, b, c, d], dim=1) '''
                bboxes = masks_to_boxes(masks).to(device)

                if epoch < 3:  # won't be used
                    loss = torch.nn.functional.mse_loss(outputs, bboxes)
                    '''
                    loss = (generalized_box_iou_loss if config.loss_type == 'general_iou' else complete_box_iou_loss)(outputs, bboxes, reduction='mean') - (
                        torch.sum(outputs[outputs < 0])) / config.batch_size +torch.sum(outputs[outputs>1])/config.batch_size \
                           + torch.mean(
                        torch.max(torch.stack((torch.zeros_like(outputs[:, 0]), outputs[:, 0] - outputs[:, 2])),
                                  keepdim=True, dim=0)[0]) + \
                           torch.mean(
                               torch.max(torch.stack((torch.zeros_like(outputs[:, 0]), outputs[:, 1] - outputs[:, 3])),
                                         keepdim=True, dim=0)[0])'''
                else:
                    if config.loss_type == 'general_iou':
                        loss = generalized_box_iou_loss(outputs, bboxes, reduction='mean')
                    elif config.loss_type == 'distance_iou':
                        loss = distance_box_iou_loss(outputs, bboxes, reduction='mean')
                    elif config.loss_type == 'mse':
                        loss = torch.nn.functional.mse_loss(outputs, bboxes, reduction='mean')


                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                if (idx + 1) % int(5 * (64 / batch_size)) == 0:
                    wandb.log({'epoch': epoch, 'loss': train_loss / (idx + 1)})
                    image = images[0].detach().cpu() * 255
                    image = image.to(torch.uint8)
                    if (idx + 1) % int(10 * (64 / batch_size)) == 0:
                        try:
                            img = torchvision.transforms.ToPILImage()(torchvision.utils.draw_bounding_boxes(image,
                                                                                                            torch.stack((
                                                                                                                        outputs[
                                                                                                                            0].clone().to(
                                                                                                                            torch.uint8),
                                                                                                                        bboxes[
                                                                                                                            0].clone().to(
                                                                                                                            torch.uint8))), colors=['green', 'red']))
                        except ValueError:
                            print(outputs[0].clone().to(torch.uint8),bboxes[0].clone().to(torch.uint8))
                            img = torchvision.transforms.ToPILImage()(image)
                        wandb.log({'epoch': epoch, 'train_image': wandb.Image(img)})
            scheduler.step(epoch=epoch)
            train_loss /= len(train_loader)

            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for idx, (images, masks) in tqdm(enumerate(val_loader)):
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)

                    '''a = torch.clamp(outputs[:, 0], max=outputs[:, 2])
                    b = torch.clamp(outputs[:, 1], max=outputs[:, 3])
                    c = torch.clamp(outputs[:, 2], min=outputs[:, 0])
                    d = torch.clamp(outputs[:, 3], min=outputs[:, 1])
                    outputs = torch.stack([a, b, c, d], dim=1)'''
                    bboxes = masks_to_boxes(masks)


                    bboxes.to(device)
                    loss = distance_box_iou_loss(outputs*126, bboxes, reduction='mean')

                    image = images[0].detach().cpu() * 255
                    image = image.to(torch.uint8)
                    val_loss += loss.item()

                    try:
                        outputs *= 126
                        print(outputs[0].clone().to(torch.uint8), bboxes[0].clone().to(torch.uint8))

                        img = torchvision.transforms.ToPILImage()(torchvision.utils.draw_bounding_boxes(image,
                                                                                                        torch.stack((
                                                                                                            outputs[
                                                                                                                0].clone().to(
                                                                                                                torch.uint8),
                                                                                                            bboxes[
                                                                                                                0].clone().to(
                                                                                                                torch.uint8))), colors=['green', 'red']))
                    except ValueError:
                        print(outputs[0].clone().to(torch.uint8), bboxes[0].clone().to(torch.uint8))
                        img = torchvision.transforms.ToPILImage()(image)

                    if epoch > 5:
                        if (int(torch.sum(outputs[outputs < 0])) < -1) or (int(torch.sum(outputs[outputs > 126]) > 1)):
                            print(f'early stopping - wrong outputted dims {outputs[outputs < 0]}, {outputs[outputs > 126]}')
                            wrong_output_dim = True
                            print(outputs[:], bboxes[:])
                            break
            val_loss /= len(val_loader)
            wandb.log({'epoch': epoch, 'val_loss': val_loss, 'val_image': wandb.Image(img)})
            if wrong_output_dim:
                break
            if val_loss < best_val_loss:
                tolerance = 5
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model_4.pt')
                tqdm.write(f'Saving the best model')
            else:
                tolerance -= 1
                if tolerance <= 0:
                    print('early stopping')
                    wandb.log({'val_loss': best_val_loss})
                    break


            # Print progress
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


'''sweep_config = {'method': 'random',
                'name': 'sweep_1',
                'program': 'bbox_train.py',
                'metric': {
                    'goal': 'minimize',
                    'name': 'val_loss'
                    },
                'parameters': {
                    'batch_size': {'min': 1, 'max': 80, 'distribution': 'int_uniform'},
                    'lr': {'max': 0.00001, 'min':  0.0000001},
                    'base_dim': {'min': 16, 'max': 80},
                    'dropout': {'min': 0.01, 'max': 0.5},
                    'batch_norm': {'values': [True, False]},
                    'normalize_images': {'values': [True, False]},
                    'loss_type':{'values': ['general_iou', 'distance_iou', 'mse'], 'probabilities': [0.3, 0.4, 0.3]},
                    'decay': {'max': 1.0, 'min':  0.7},
                    },

                }'''
#sweep_id = wandb.sweep(sweep=sweep_config, project="sweep_1")

#wandb.agent(sweep_id, function=main, count=100)

main()




