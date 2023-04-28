
from seg_model import SegmentationModel

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import torch.optim as optim
    from my_dataset import MyDataset
    from glob import glob
    import os
    from tqdm import tqdm
    import torchvision

    torchvision.disable_beta_transforms_warning()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    images_path = r'C:\my files\REFUGE'
    train_images_path = 'Training400/**/*.jpg'
    train_masks_path = 'Annotation-Training400/Disc_Cup_Masks/**/*.bmp'
    val_images_path = 'REFUGE-Validation400/**/*.jpg'
    val_masks_path = 'REFUGE-Validation400-GT/**/*.bmp'

    train_images = sorted(glob(os.path.join(images_path, train_images_path), recursive=True))
    train_masks = sorted(glob(os.path.join(images_path, train_masks_path), recursive=True))
    val_images = sorted(glob(os.path.join(images_path, val_images_path), recursive=True))
    val_masks = sorted(glob(os.path.join(images_path, val_masks_path), recursive=True))

    path = r'C:\my files\REFUGE\training'

    train_masks = sorted(glob(os.path.join(path, r'masks/**/*.bmp'), recursive=True))
    train_images = sorted(glob(os.path.join(path, r'input/**/*.jpg'), recursive=True))

    path = r'C:\my files\REFUGE\validation'

    val_masks = sorted(glob(os.path.join(path, r'masks/**/*.bmp'), recursive=True))
    val_images = sorted(glob(os.path.join(path, r'input/**/*.jpg'), recursive=True))

    train_dataset = MyDataset(train_images, train_masks)
    val_dataset = MyDataset(val_images, val_masks)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    import torch.optim as optim

    # Define your model
    model = SegmentationModel(in_channels=3, out_channels=3).to(device)

    # Define your loss function
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., 10., 10.])).to(device)

    # Define your optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define number of epochs
    num_epochs = 10

    # Train the model
    torch.cuda.empty_cache()
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for idx, (images, masks) in tqdm(enumerate(train_loader)):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            if idx % 20 == 0:
                tqdm.write(str(loss.item()))
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
    
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
            val_loss /= len(val_loader)

        # Print progress
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # %%
    torch.cuda.empty_cache()
