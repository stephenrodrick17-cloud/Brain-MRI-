"""
3D Volume Prediction from Single 2D Medical Images
====================================================
Uses 3D CNN to predict volumetric data from 2D slices

Architecture: 3D Encoder-Decoder with Attention
- Input: 2D grayscale image (1, H, W)
- Output: 3D volume (D, H, W) where D is depth dimension
- Can be trained with paired 2D-3D data

Features:
- 3D convolutions for spatial reasoning
- Skip connections for detail preservation
- Attention gates for relevant depth layers
- Supports multi-resolution predictions

Usage:
    from model_3d_prediction import Volume3DPredictor
    model = Volume3DPredictor(depth=32)
    volume = model(image_2d)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class Conv3DBlock(nn.Module):
    """3D Convolution block with batch normalization"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class Attention3DGate(nn.Module):
    """3D Attention mechanism"""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(Attention3DGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, 1, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, 1, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, 1, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Slice2Volume3DCNN(nn.Module):
    """
    3D CNN for predicting 3D volumes from 2D slices
    
    Architecture:
    - Converts 2D input to 3D feature space
    - Learns depth relationships through 3D convolutions
    - Attention gates for focusing on relevant layers
    - Deep supervision for better training
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, 
                 depth: int = 16, base_features: int = 32):
        """
        Args:
            in_channels: Input channels (1 for grayscale)
            out_channels: Output channels (1 for single volume)
            depth: Depth dimension of output volume (D)
            base_features: Base feature channels
        """
        super(Slice2Volume3DCNN, self).__init__()
        
        self.depth = depth
        self.base_features = base_features
        
        # ==================== ENCODER ====================
        # First, convert 2D to 3D by expanding depth dimension
        self.slice_to_3d = nn.Sequential(
            nn.Conv2d(in_channels, base_features, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Encoder - Downsampling path
        self.enc1 = Conv3DBlock(base_features, base_features)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = Conv3DBlock(base_features, base_features * 2)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = Conv3DBlock(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = Conv3DBlock(base_features * 4, base_features * 8)
        
        # ==================== DECODER ====================
        # Decoder - Upsampling path
        self.upconv3 = nn.ConvTranspose3d(base_features * 8, base_features * 4, 2, stride=2)
        self.attn3 = Attention3DGate(base_features * 4, base_features * 4, base_features * 2)
        self.dec3 = Conv3DBlock(base_features * 8, base_features * 4)
        
        self.upconv2 = nn.ConvTranspose3d(base_features * 4, base_features * 2, 2, stride=2)
        self.attn2 = Attention3DGate(base_features * 2, base_features * 2, base_features)
        self.dec2 = Conv3DBlock(base_features * 4, base_features * 2)
        
        self.upconv1 = nn.ConvTranspose3d(base_features * 2, base_features, 2, stride=2)
        self.attn1 = Attention3DGate(base_features, base_features, base_features // 2)
        self.dec1 = Conv3DBlock(base_features * 2, base_features)
        
        # Final output
        self.final = nn.Conv3d(base_features, out_channels, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Deep supervision (auxiliary outputs)
        self.deep_sup3 = nn.Conv3d(base_features * 4, out_channels, 1)
        self.deep_sup2 = nn.Conv3d(base_features * 2, out_channels, 1)
        self.deep_sup1 = nn.Conv3d(base_features, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) - 2D image batch
            
        Returns:
            (B, 1, D, H, W) - 3D volume predictions
        """
        B, C, H, W = x.shape
        
        # Convert 2D to 3D by expanding along depth
        # (B, 1, H, W) -> (B, 64, D, H, W)
        x_2d = self.slice_to_3d(x)  # (B, 64, H, W)
        
        # Expand depth dimension
        x_3d = x_2d.unsqueeze(2).expand(B, self.base_features, self.depth, H, W)
        
        # Add positional encoding for depth
        depth_pos = self._positional_depth_encoding(B, self.depth, self.base_features, x_3d.device)
        x_3d = x_3d + depth_pos
        
        # ==================== ENCODING ====================
        # Encoder block 1
        enc1 = self.enc1(x_3d)  # (B, 64, D, H, W)
        pool1 = self.pool1(enc1)  # (B, 64, D/2, H/2, W/2)
        
        # Encoder block 2
        enc2 = self.enc2(pool1)  # (B, 128, D/2, H/2, W/2)
        pool2 = self.pool2(enc2)  # (B, 128, D/4, H/4, W/4)
        
        # Encoder block 3
        enc3 = self.enc3(pool2)  # (B, 256, D/4, H/4, W/4)
        pool3 = self.pool3(enc3)  # (B, 256, D/8, H/8, W/8)
        
        # Bottleneck
        bottleneck = self.bottleneck(pool3)  # (B, 512, D/8, H/8, W/8)
        
        # ==================== DECODING ====================
        # Decoder block 3
        upconv3 = self.upconv3(bottleneck)  # (B, 256, D/4, H/4, W/4)
        attn3 = self.attn3(upconv3, enc3)
        dec3 = torch.cat([attn3, upconv3], dim=1)
        dec3 = self.dec3(dec3)  # (B, 256, D/4, H/4, W/4)
        
        # Decoder block 2
        upconv2 = self.upconv2(dec3)  # (B, 128, D/2, H/2, W/2)
        attn2 = self.attn2(upconv2, enc2)
        dec2 = torch.cat([attn2, upconv2], dim=1)
        dec2 = self.dec2(dec2)  # (B, 128, D/2, H/2, W/2)
        
        # Decoder block 1
        upconv1 = self.upconv1(dec2)  # (B, 64, D, H, W)
        attn1 = self.attn1(upconv1, enc1)
        dec1 = torch.cat([attn1, upconv1], dim=1)
        dec1 = self.dec1(dec1)  # (B, 64, D, H, W)
        
        # Final output
        output = self.final(dec1)  # (B, 1, D, H, W)
        output = self.sigmoid(output)
        
        return output
    
    @staticmethod
    def _positional_depth_encoding(batch_size: int, depth: int, channels: int, device: torch.device) -> torch.Tensor:
        """
        Create positional encoding for depth dimension
        Helps model understand depth ordering
        """
        # Sinusoidal positional encoding
        depth_pos = torch.arange(depth, dtype=torch.float32, device=device)
        depth_pos = depth_pos / depth  # Normalize to 0-1
        
        # Create positional embeddings with correct number of channels
        freqs = torch.exp(torch.arange(0, channels, 2, dtype=torch.float32, device=device) 
                         * (-np.log(10000) / channels))
        
        # Compute sin and cos embeddings
        pos_enc = torch.zeros(depth, channels, device=device)
        pos_enc[:, 0::2] = torch.sin(depth_pos.unsqueeze(1) * freqs)
        if channels % 2 == 1:
            pos_enc[:, 1::2] = torch.cos(depth_pos.unsqueeze(1) * freqs[:-1])
        else:
            pos_enc[:, 1::2] = torch.cos(depth_pos.unsqueeze(1) * freqs)
        
        # Expand to batch size
        pos_enc = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)
        # (B, D, channels) -> (B, channels, D, 1, 1)
        pos_enc = pos_enc.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        
        return pos_enc * 0.1  # Scale down to avoid overwhelming features


class Volume3DPredictor:
    """
    High-level interface for 3D volume prediction
    """
    
    def __init__(self, model_path: str = None, depth: int = 32, 
                 device: str = 'cpu'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model (optional)
            depth: Output volume depth
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.depth = depth
        self.model = None
        self.model_loaded = False
        self.model = Slice2Volume3DCNN(depth=depth).to(self.device)
        
        if model_path:
            self.load_model(model_path)
        
        self.model.eval()
    
    def load_model(self, model_path: str):
        """Load trained weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"[OK] Loaded model from {model_path}")
    
    def predict(self, image_2d: np.ndarray) -> np.ndarray:
        """
        Predict 3D volume from 2D image (Optimized for speed)
        
        Args:
            image_2d: 2D image array (H, W) or (H, W, 1)
            
        Returns:
            3D volume (D, H, W)
        """
        # Preprocess
        if len(image_2d.shape) == 3:
            image_2d = image_2d[:, :, 0]
        
        # Normalize
        img_min = image_2d.min()
        img_max = image_2d.max()
        image_2d = (image_2d - img_min) / (img_max - img_min + 1e-8)
        
        # Convert to tensor with minimal overhead
        image_tensor = torch.from_numpy(image_2d.astype(np.float32))
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Predict with no gradient computation
        with torch.no_grad():
            volume_3d = self.model(image_tensor)
        
        # Extract and convert to numpy (move to CPU first for speed)
        volume_3d = volume_3d.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        
        return volume_3d
    
    def predict_batch(self, images_2d: np.ndarray) -> np.ndarray:
        """
        Predict 3D volumes for batch of 2D images
        
        Args:
            images_2d: Batch of 2D images (B, H, W)
            
        Returns:
            Batch of 3D volumes (B, D, H, W)
        """
        B, H, W = images_2d.shape
        
        # Normalize
        images_2d = (images_2d - images_2d.min(axis=(1, 2), keepdims=True)) / \
                    (images_2d.max(axis=(1, 2), keepdims=True) - 
                     images_2d.min(axis=(1, 2), keepdims=True) + 1e-8)
        
        # Convert to tensor
        images_tensor = torch.from_numpy(images_2d).float()
        images_tensor = images_tensor.unsqueeze(1).to(self.device)
        
        # Predict
        with torch.no_grad():
            volumes_3d = self.model(images_tensor)
        
        # Convert to numpy
        volumes_3d = volumes_3d.squeeze(1).cpu().numpy()
        
        return volumes_3d


if __name__ == '__main__':
    print("3D Volume Prediction Model")
    print("="*50)
    
    # Example usage
    print("\nModel Architecture:")
    model = Slice2Volume3DCNN(depth=32)
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,}")
    
    print("\n" + "="*50)
    print("Example inference:")
    print("```python")
    print("from model_3d_prediction import Volume3DPredictor")
    print("predictor = Volume3DPredictor(depth=32)")
    print("volume_3d = predictor.predict(image_2d)")
    print("print(volume_3d.shape)  # (32, H, W)")
    print("```")
