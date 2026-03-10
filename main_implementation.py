"""
Medical Image Segmentation AI System
=====================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json

# Random seeds
np.random.seed(42)
torch.manual_seed(42)


class ConvBlock(nn.Module):
    """Double convolution block"""
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention mechanism"""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """Attention U-Net Architecture"""
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super(AttentionUNet, self).__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)
        
        # Decoder with Attention
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.dec4 = ConvBlock(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec3 = ConvBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec2 = ConvBlock(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec1 = ConvBlock(128, 64)
        
        # Final output
        self.out = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder with Attention
        u4 = self.up4(b)
        e4_att = self.att4(u4, e4)
        d4 = self.dec4(torch.cat([u4, e4_att], dim=1))
        
        u3 = self.up3(d4)
        e3_att = self.att3(u3, e3)
        d3 = self.dec3(torch.cat([u3, e3_att], dim=1))
        
        u2 = self.up2(d3)
        e2_att = self.att2(u2, e2)
        d2 = self.dec2(torch.cat([u2, e2_att], dim=1))
        
        u1 = self.up1(d2)
        e1_att = self.att1(u1, e1)
        d1 = self.dec1(torch.cat([u1, e1_att], dim=1))
        
        # Output
        out = self.out(d1)
        return out


class MedicalImageDataset(Dataset):
    """Medical image dataset"""
    def __init__(self, images: np.ndarray, masks: np.ndarray, augment: bool = False):
        self.images = images
        self.masks = masks
        self.augment = augment
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)
        mask = self.masks[idx].astype(np.float32)
        
        # Normalization
        image = (image - np.mean(image)) / (np.std(image) + 1e-8)
        
        # Intensity clipping
        image = np.clip(image, -3, 3)
        
        # Add channel dimension
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=0)
        
        return torch.FloatTensor(image), torch.FloatTensor(mask)


def generate_synthetic_medical_data(num_samples: int = 500, img_size: int = 256):
    """Generate synthetic medical data"""
    images = []
    masks = []
    
    for i in range(num_samples):
        # Create base image with tissue-like texture
        img = np.random.randn(img_size, img_size) * 0.1 + 0.5
        mask = np.zeros((img_size, img_size))
        
        # Add anatomical structures
        num_structures = np.random.randint(2, 5)
        for j in range(num_structures):
            center_x = np.random.randint(img_size // 4, 3 * img_size // 4)
            center_y = np.random.randint(img_size // 4, 3 * img_size // 4)
            radius_x = np.random.randint(20, 60)
            radius_y = np.random.randint(20, 60)
            intensity = np.random.uniform(0.6, 0.9)
            
            y, x = np.ogrid[:img_size, :img_size]
            ellipse = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2 <= 1
            
            img[ellipse] = intensity + np.random.randn(np.sum(ellipse)) * 0.05
            mask[ellipse] = 1.0
        
        # Add noise
        img += np.random.randn(img_size, img_size) * 0.05
        
        images.append(img)
        masks.append(mask)
    
    return np.array(images), np.array(masks)


class DiceLoss(nn.Module):
    """Dice Loss"""
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined Dice + BCE Loss"""
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


def calculate_dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculate Dice similarity coefficient"""
    pred = (torch.sigmoid(pred) > threshold).float()
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-8)
    
    return dice.item()


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculate Intersection over Union"""
    pred = (torch.sigmoid(pred) > threshold).float()
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-8)
    
    return iou.item()


class Trainer:
    """Training pipeline"""
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: str = 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_dice': [], 'val_dice': [],
            'train_iou': [], 'val_iou': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stop_patience = 15
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0
        
        for images, masks in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_dice += calculate_dice_score(outputs, masks)
            epoch_iou += calculate_iou(outputs, masks)
        
        num_batches = len(self.train_loader)
        return {
            'loss': epoch_loss / num_batches,
            'dice': epoch_dice / num_batches,
            'iou': epoch_iou / num_batches
        }
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                epoch_loss += loss.item()
                epoch_dice += calculate_dice_score(outputs, masks)
                epoch_iou += calculate_iou(outputs, masks)
        
        num_batches = len(self.val_loader)
        return {
            'loss': epoch_loss / num_batches,
            'dice': epoch_dice / num_batches,
            'iou': epoch_iou / num_batches
        }
    
    def train(self, num_epochs: int = 50):
        """Full training loop"""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            self.scheduler.step(val_metrics['loss'])
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_dice'].append(train_metrics['dice'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['train_iou'].append(train_metrics['iou'])
            self.history['val_iou'].append(val_metrics['iou'])
            
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Dice: {train_metrics['dice']:.4f} | IoU: {train_metrics['iou']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f}")
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
                print("  → Model saved!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stop_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
            
            print("-" * 60)
        
        print("\nTraining completed!")
        return self.history


def main():
    """Main execution"""
    print("=" * 70)
    print("MEDICAL IMAGE SEGMENTATION AI SYSTEM")
    print("=" * 70)
    
    config = {
        'num_samples': 500,
        'img_size': 256,
        'batch_size': 8,
        'num_epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Step 1: Generate Data
    print("\n" + "-" * 70)
    print("STEP 1: Data Generation and Preprocessing")
    print("-" * 70)
    
    images, masks = generate_synthetic_medical_data(
        num_samples=config['num_samples'],
        img_size=config['img_size']
    )
    print(f"✓ Generated {len(images)} synthetic medical images")
    print(f"  Image shape: {images.shape}")
    print(f"  Mask shape: {masks.shape}")
    
    train_images, temp_images, train_masks, temp_masks = train_test_split(
        images, masks, test_size=0.3, random_state=42
    )
    val_images, test_images, val_masks, test_masks = train_test_split(
        temp_images, temp_masks, test_size=0.5, random_state=42
    )
    
    print(f"\n✓ Data split completed:")
    print(f"  Training set: {len(train_images)} samples")
    print(f"  Validation set: {len(val_images)} samples")
    print(f"  Test set: {len(test_images)} samples")
    
    # Create datasets and loaders
    train_dataset = MedicalImageDataset(train_images, train_masks, augment=True)
    val_dataset = MedicalImageDataset(val_images, val_masks, augment=False)
    test_dataset = MedicalImageDataset(test_images, test_masks, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # Step 2: Model Initialization
    print("\n" + "-" * 70)
    print("STEP 2: Model Architecture Initialization")
    print("-" * 70)
    
    model = AttentionUNet(in_channels=1, num_classes=1)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Attention U-Net initialized")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Step 3: Training
    print("\n" + "-" * 70)
    print("STEP 3: Model Training")
    print("-" * 70)
    
    trainer = Trainer(model, train_loader, val_loader, device=config['device'])
    history = trainer.train(num_epochs=config['num_epochs'])
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    print("\n✓ Best model loaded for evaluation")
    
    # Step 4: Save Results
    print("\n" + "-" * 70)
    print("STEP 4: Saving Results")
    print("-" * 70)
    
    results = {
        'config': config,
        'model_architecture': 'Attention U-Net',
        'total_parameters': total_params,
        'training_history': history
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("✓ Results saved to 'results.json'")
    print("✓ Model saved to 'best_model.pth'")
    
    # Final Summary
    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)
    print(f"\n✓ Training completed successfully")
    print(f"✓ Best validation Dice score: {max(history['val_dice']):.4f}")
    print(f"✓ Final test Dice score: {calculate_dice_score(model(torch.FloatTensor(test_images[:1]).to(config['device'])), torch.FloatTensor(test_masks[:1]).to(config['device'])):.4f}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
