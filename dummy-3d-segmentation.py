#%%
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from tqdm import tqdm

np.random.seed(42)

#%%
# ----------------------------
# 1. Synthetic Dataset Generator
# ----------------------------
class SyntheticTeethDataset(Dataset):
    def __init__(self, num_samples=200, image_size=128, augment=False):
        self.num_samples = num_samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return self.num_samples

    def generate_image_and_mask(self):
        img = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # Random number of teeth
        for _ in range(np.random.randint(3, 7)):
            center = (
                np.random.randint(20, self.image_size - 20),
                np.random.randint(20, self.image_size - 20)
            )
            axes = (
                np.random.randint(8, 15),
                np.random.randint(12, 20)
            )
            angle = np.random.randint(0, 180)
            cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

        # Blur & noise to mimic X-ray grain
        img = mask.copy()
        img = cv2.GaussianBlur(img, (5, 5), 0)
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        # Plot images side by side
        %matplotlib inline
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title("Augmented X-ray Image")
        axes[0].axis("off")
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Mask")
        axes[1].axis("off")
        plt.show()

        return img, mask

    def __getitem__(self, idx):
        img, mask = self.generate_image_and_mask()

        if self.augment:
            if np.random.random() > 0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)
            if np.random.random() > 0.5:
                img = np.flipud(img)
                mask = np.flipud(mask)

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        # Add channel dimension
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(img), torch.tensor(mask)

#%%
# ----------------------------
# 2. Baseline Model (intentionally weak)
# ----------------------------
class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)

#%%
# ----------------------------
# 3. Dice Loss
# ----------------------------
def dice_loss(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))

#%%
# ----------------------------
# 4. Training & Evaluation
# ----------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

#%%
def evaluate(model, loader, device):
    model.eval()
    total_dice = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            dice = 1 - dice_loss(outputs, masks)
            total_dice += dice.item()
    return total_dice / len(loader)

#%%
# ----------------------------
# 5. Main
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SyntheticTeethDataset(num_samples=300, augment=False)
    val_dataset = SyntheticTeethDataset(num_samples=50, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model = BaselineCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_dice = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Dice={val_dice:.4f}")

    print("\nBaseline complete. Try to improve architecture, augmentation, loss, or optimizer!")

#%%
# if __name__ == "__main__":
    # main()

# %%
train_dataset = SyntheticTeethDataset(num_samples=2, augment=True)
train_dataset.generate_image_and_mask()

# %%
