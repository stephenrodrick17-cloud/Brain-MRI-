#!/usr/bin/env python3
"""
Simple 2D Segmentation Model Trainer
Creates best_model.pth for the 3D reconstruction pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os

print("="*70)
print("  Training 2D Segmentation Model")
print("="*70)

# Check if model exists
if os.path.exists('best_model.pth'):
    print("\n[OK] best_model.pth already exists!")
    print("     Skipping training...\n")
else:
    print("\n[INFO] Creating synthetic training data...")
    
    # Create synthetic data
    num_samples = 100
    img_size = 256
    
    images = np.zeros((num_samples, img_size, img_size), dtype=np.float32)
    masks = np.zeros((num_samples, img_size, img_size), dtype=np.float32)
    
    np.random.seed(42)
    
    for i in range(num_samples):
        # Random brain-like structure
        img = np.zeros((img_size, img_size), dtype=np.float32)
        
        # Add some circles for brain structure
        y, x = np.ogrid[-img_size//2:img_size//2, -img_size//2:img_size//2]
        
        # Center structure
        r1 = np.random.randint(30, 60)
        mask1 = (x**2 + y**2) <= (r1**2)
        img[mask1] = 0.8 + np.random.rand() * 0.2
        
        # Outer structure
        r2 = np.random.randint(80, 120)
        mask2 = (x**2 + y**2) <= (r2**2)
        img[mask2] = 0.5 + np.random.rand() * 0.3
        
        # Add noise
        img += np.random.randn(img_size, img_size) * 0.1
        img = np.clip(img, 0, 1)
        
        # Create mask
        mask = np.zeros((img_size, img_size), dtype=np.float32)
        mask[mask2] = 1.0
        
        images[i] = img
        masks[i] = mask
    
    print(f"[OK] Generated {num_samples} synthetic images")
    print(f"     Shape: {images.shape}")
    
    # Create a simple CNN-based model
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            
            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, 5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, 5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            
            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(16, 8, 3, padding=1),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(8, 1, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x
    
    print("[INFO] Building model...")
    device = torch.device('cpu')
    model = SimpleCNN().to(device)
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model created with {params:,} parameters")
    
    # Create data loaders
    class MedicalImageDataset(Dataset):
        def __init__(self, images, masks):
            self.images = torch.from_numpy(images).unsqueeze(1)
            self.masks = torch.from_numpy(masks).unsqueeze(1)
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            return self.images[idx], self.masks[idx]
    
    dataset = MedicalImageDataset(images, masks)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print(f"[OK] Data loaders created")
    print(f"     Training: {len(train_dataset)} samples")
    print(f"     Validation: {len(val_dataset)} samples")
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    num_epochs = 10
    
    print(f"\n[INFO] Starting training for {num_epochs} epochs...")
    
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for images_batch, masks_batch in train_loader:
            images_batch = images_batch.to(device)
            masks_batch = masks_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(images_batch)
            loss = criterion(outputs, masks_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images_batch, masks_batch in val_loader:
                images_batch = images_batch.to(device)
                masks_batch = masks_batch.to(device)
                outputs = model(images_batch)
                loss = criterion(outputs, masks_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
            print(f"   [OK] Model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping at epoch {epoch+1}")
                break
    
    print("\n[OK] Training completed!")
    print("[OK] Model saved to 'best_model.pth'")

print("\nNext step: Run python quickstart_3d.py\n")
