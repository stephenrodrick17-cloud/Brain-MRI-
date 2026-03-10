"""
Training Script for 3D Volume Prediction from 2D Slices
========================================================
Trains the 3D CNN to predict volumetric data from single 2D images

Features:
- Support for synthetic and real paired 2D-3D data
- Automatic synthetic data generation
- Advanced loss functions (L1, SSIM, perceptual)
- Validation and early stopping
- Model checkpointing

Usage:
    python train_3d_predictor.py --data-dir ./data --epochs 100 --batch-size 8
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from model_3d_prediction import Slice2Volume3DCNN


class SyntheticPaired3DDataset(Dataset):
    """
    Generate synthetic paired 2D-3D data
    Creates realistic medical image 2D slices and their corresponding 3D volumes
    """
    
    def __init__(self, num_samples: int = 500, img_size: int = 256, 
                 depth: int = 16, mode: str = 'train'):
        """
        Args:
            num_samples: Number of synthetic samples to generate
            img_size: Image dimension (will be img_size x img_size)
            depth: Depth of 3D volume
            mode: 'train', 'val', or 'test'
        """
        self.num_samples = num_samples
        self.img_size = img_size
        self.depth = depth
        self.mode = mode
        
        # Generate synthetic data
        self.samples = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> List[Dict]:
        """Generate synthetic paired 2D-3D data"""
        samples = []
        
        for idx in range(self.num_samples):
            # Generate synthetic 3D volume
            volume_3d = self._create_synthetic_volume()
            
            # Extract middle slice as 2D image
            middle_slice = self.depth // 2
            image_2d = volume_3d[middle_slice]
            
            # Add noise to 2D slice
            noise = np.random.normal(0, 0.05, image_2d.shape)
            image_2d = np.clip(image_2d + noise, 0, 1)
            
            samples.append({
                'image_2d': image_2d.astype(np.float32),
                'volume_3d': volume_3d.astype(np.float32)
            })
        
        return samples
    
    def _create_synthetic_volume(self) -> np.ndarray:
        """
        Create realistic synthetic 3D medical image volume
        Simulates anatomical structures at different depths
        """
        volume = np.zeros((self.depth, self.img_size, self.img_size), dtype=np.float32)
        
        # Create multiple Gaussian blobs at different depths (simulating anatomy)
        num_structures = np.random.randint(3, 8)
        
        for struct_idx in range(num_structures):
            # Random center and size
            center_d = np.random.randint(0, self.depth)
            center_h = np.random.randint(30, self.img_size - 30)
            center_w = np.random.randint(30, self.img_size - 30)
            radius = np.random.randint(10, 40)
            intensity = np.random.uniform(0.5, 1.0)
            
            # Create Gaussian blob
            d_idx, h_idx, w_idx = np.ogrid[:self.depth, :self.img_size, :self.img_size]
            
            # Distance from center
            dist_d = (d_idx - center_d) ** 2
            dist_h = (h_idx - center_h) ** 2
            dist_w = (w_idx - center_w) ** 2
            
            # Gaussian falloff
            blob = intensity * np.exp(-(dist_d + dist_h + dist_w) / (2 * radius ** 2))
            volume = np.maximum(volume, blob)
        
        # Add depth variation (simulate tissue density changes)
        depth_gradient = np.linspace(0.3, 1.0, self.depth)[:, np.newaxis, np.newaxis]
        volume = volume * depth_gradient
        
        # Add realistic noise
        noise = np.random.normal(0, 0.1, volume.shape)
        volume = np.clip(volume + noise, 0, 1)
        
        # Smooth for realism
        from scipy.ndimage import gaussian_filter
        volume = gaussian_filter(volume, sigma=1.5)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        return volume
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        image_2d = torch.from_numpy(sample['image_2d']).unsqueeze(0)  # (1, H, W)
        volume_3d = torch.from_numpy(sample['volume_3d']).unsqueeze(0)  # (1, D, H, W)
        
        return image_2d, volume_3d


class RealPaired3DDataset(Dataset):
    """
    Load real paired 2D-3D medical imaging data
    
    Expected directory structure:
    data_dir/
        images_2d/
            *.jpg or *.png
        volumes_3d/
            *.npy (shape: D, H, W)
    """
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to dataset directory
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / 'images_2d'
        self.volumes_dir = self.data_dir / 'volumes_3d'
        
        # Find matching pairs
        self.image_files = sorted(self.images_dir.glob('*.[jp][pn][g]'))
        
        if not self.image_files:
            raise ValueError(f"No 2D images found in {self.images_dir}")
        
        print(f"Found {len(self.image_files)} image-volume pairs")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image
        
        # Load 2D image
        img_path = self.image_files[idx]
        image_2d = Image.open(img_path).convert('L')
        image_2d = np.array(image_2d, dtype=np.float32) / 255.0
        
        # Load corresponding 3D volume
        vol_path = self.volumes_dir / (img_path.stem + '.npy')
        volume_3d = np.load(vol_path).astype(np.float32)
        
        # Normalize if needed
        if volume_3d.max() > 1.0:
            volume_3d = volume_3d / 255.0
        
        image_2d = torch.from_numpy(image_2d).unsqueeze(0)
        volume_3d = torch.from_numpy(volume_3d).unsqueeze(0)
        
        return image_2d, volume_3d


class VolumeSSIMLoss(nn.Module):
    """SSIM Loss for 3D volumes"""
    
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss"""
        # Simplified SSIM: correlation-based
        x_flat = x.view(x.size(0), -1)
        y_flat = y.view(y.size(0), -1)
        
        # Compute correlation
        x_mean = x_flat.mean(dim=1, keepdim=True)
        y_mean = y_flat.mean(dim=1, keepdim=True)
        
        x_centered = x_flat - x_mean
        y_centered = y_flat - y_mean
        
        cov = (x_centered * y_centered).mean(dim=1)
        x_var = (x_centered ** 2).mean(dim=1)
        y_var = (y_centered ** 2).mean(dim=1)
        
        ssim = (2 * cov + 1e-6) / (x_var + y_var + 1e-6)
        
        return 1.0 - ssim.mean()


class Trainer:
    """Training orchestrator"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu',
                 lr: float = 1e-3, weight_decay: float = 1e-5):
        """
        Initialize trainer
        
        Args:
            model: 3D volume prediction model
            device: 'cpu' or 'cuda'
            lr: Learning rate
            weight_decay: L2 regularization
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = VolumeSSIMLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
        self.best_val_loss = float('inf')
        self.best_model_path = None
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for images_2d, volumes_3d in tqdm(train_loader, desc='Training'):
            images_2d = images_2d.to(self.device)
            volumes_3d = volumes_3d.to(self.device)
            
            # Forward pass
            predictions = self.model(images_2d)
            
            # Multi-loss combination
            l1 = self.l1_loss(predictions, volumes_3d)
            mse = self.mse_loss(predictions, volumes_3d)
            ssim = self.ssim_loss(predictions, volumes_3d)
            
            # Combined loss
            loss = 0.5 * l1 + 0.3 * mse + 0.2 * ssim
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validation pass"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images_2d, volumes_3d in tqdm(val_loader, desc='Validation'):
                images_2d = images_2d.to(self.device)
                volumes_3d = volumes_3d.to(self.device)
                
                predictions = self.model(images_2d)
                
                l1 = self.l1_loss(predictions, volumes_3d)
                mse = self.mse_loss(predictions, volumes_3d)
                ssim = self.ssim_loss(predictions, volumes_3d)
                
                loss = 0.5 * l1 + 0.3 * mse + 0.2 * ssim
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int, save_dir: str = './checkpoints'):
        """
        Complete training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Training 3D Volume Predictor")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"{'='*60}\n")
        
        patience = 20
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Learning rate step
            self.scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                self.best_model_path = save_dir / f'best_3d_model_epoch{epoch+1}.pth'
                self.save_checkpoint(self.best_model_path)
                print(f"  ✓ Best model saved: {self.best_model_path.name}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n✓ Early stopping at epoch {epoch+1}")
                    break
        
        # Save final model
        final_path = save_dir / 'final_3d_model.pth'
        self.save_checkpoint(final_path)
        
        # Save history
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ Training complete!")
        print(f"  Best model: {self.best_model_path}")
        print(f"  Best val loss: {self.best_val_loss:.6f}")
        print(f"{'='*60}\n")
        
        # Plot training history
        self._plot_history(save_dir / 'training_history.png')
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint"""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'history': self.history
        }, path)
    
    def _plot_history(self, save_path: Path):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.history['lr'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✓ Saved training history plot: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train 3D Volume Predictor')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Path to paired 2D-3D data directory')
    parser.add_argument('--synthetic', action='store_true', default=True,
                       help='Use synthetic data if no real data provided')
    parser.add_argument('--num-samples', type=int, default=500,
                       help='Number of synthetic samples')
    parser.add_argument('--img-size', type=int, default=256,
                       help='Image size')
    parser.add_argument('--depth', type=int, default=16,
                       help='Output volume depth')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device')
    parser.add_argument('--output-dir', type=str, default='./models_3d',
                       help='Output directory for models')
    
    args = parser.parse_args()
    
    # Create dataset
    if args.data_dir and Path(args.data_dir).exists():
        print(f"Loading real data from {args.data_dir}")
        dataset = RealPaired3DDataset(args.data_dir)
    else:
        print(f"Generating synthetic data ({args.num_samples} samples)...")
        dataset = SyntheticPaired3DDataset(
            num_samples=args.num_samples,
            img_size=args.img_size,
            depth=args.depth
        )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create model
    model = Slice2Volume3DCNN(depth=args.depth)
    
    # Train
    trainer = Trainer(model, device=args.device, lr=args.lr)
    trainer.train(train_loader, val_loader, num_epochs=args.epochs,
                  save_dir=args.output_dir)


if __name__ == '__main__':
    main()
