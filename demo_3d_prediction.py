"""
Demonstration of 3D CNN Volume Prediction
==========================================
Shows how the 3D CNN transforms 2D slices into 3D volumes

Features:
- Visualization of depth progression
- Comparison with synthetic data
- Inference examples
- Performance metrics

Run:
    python demo_3d_prediction.py [--model path/to/model.pth]
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import argparse

from model_3d_prediction import Volume3DPredictor, Slice2Volume3DCNN
from train_3d_predictor import SyntheticPaired3DDataset


def visualize_volume_slices(volume: np.ndarray, title: str = "3D Volume", 
                            save_path: str = None):
    """
    Visualize 3D volume from multiple angles
    
    Args:
        volume: 3D volume (D, H, W)
        title: Plot title
        save_path: Path to save figure
    """
    D, H, W = volume.shape
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 4, figure=fig)
    
    # Title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Axial slices (top to bottom)
    print(f"Visualizing volume shape: {volume.shape}")
    
    axial_indices = np.linspace(0, D-1, 4, dtype=int)
    for i, idx in enumerate(axial_indices):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(volume[idx], cmap='viridis', origin='upper')
        ax.set_title(f'Depth {idx} (Axial)', fontsize=10)
        ax.axis('off')
    
    # Coronal slices (left to right across volume)
    coronal_indices = np.linspace(0, H-1, 4, dtype=int)
    for i, idx in enumerate(coronal_indices):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(volume[:, idx, :], aspect='auto', cmap='viridis', origin='upper')
        ax.set_title(f'Position {idx} (Coronal)', fontsize=10)
        ax.axis('off')
    
    # Sagittal slices (front to back)
    sagittal_indices = np.linspace(0, W-1, 4, dtype=int)
    for i, idx in enumerate(sagittal_indices):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(volume[:, :, idx], aspect='auto', cmap='viridis', origin='upper')
        ax.set_title(f'Position {idx} (Sagittal)', fontsize=10)
        ax.axis('off')
    
    # Projections and depth map
    mip = np.max(volume, axis=0)
    min_proj = np.min(volume, axis=0)
    mean_depth = np.mean(volume, axis=0)
    std_depth = np.std(volume, axis=0)
    
    ax1 = fig.add_subplot(gs[3, 0])
    ax1.imshow(mip, cmap='hot')
    ax1.set_title('Max Intensity Projection', fontsize=10)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[3, 1])
    ax2.imshow(min_proj, cmap='cool')
    ax2.set_title('Min Intensity Projection', fontsize=10)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[3, 2])
    im = ax3.imshow(mean_depth, cmap='viridis')
    ax3.set_title('Mean Depth', fontsize=10)
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    
    ax4 = fig.add_subplot(gs[3, 3])
    im = ax4.imshow(std_depth, cmap='plasma')
    ax4.set_title('Depth Variation', fontsize=10)
    ax4.axis('off')
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    return fig


def compare_predictions(image_2d: np.ndarray, predicted_volume: np.ndarray,
                        ground_truth_volume: np.ndarray = None,
                        save_path: str = None):
    """
    Compare predicted volume with ground truth
    
    Args:
        image_2d: Input 2D image
        predicted_volume: Predicted 3D volume
        ground_truth_volume: Optional ground truth for comparison
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('3D CNN Volume Prediction Results', fontsize=14, fontweight='bold')
    
    # Input image
    axes[0, 0].imshow(image_2d, cmap='gray')
    axes[0, 0].set_title('Input 2D Slice')
    axes[0, 0].axis('off')
    
    # Predicted depth map
    pred_depth = np.mean(predicted_volume, axis=0)
    im1 = axes[0, 1].imshow(pred_depth, cmap='viridis')
    axes[0, 1].set_title('Predicted Average Depth')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Predicted MIP
    pred_mip = np.max(predicted_volume, axis=0)
    im2 = axes[0, 2].imshow(pred_mip, cmap='hot')
    axes[0, 2].set_title('Predicted Max Intensity')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2])
    
    if ground_truth_volume is not None:
        # Ground truth depth
        gt_depth = np.mean(ground_truth_volume, axis=0)
        im3 = axes[1, 0].imshow(gt_depth, cmap='viridis')
        axes[1, 0].set_title('Ground Truth Average Depth')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Difference
        diff = np.abs(pred_depth - gt_depth)
        im4 = axes[1, 1].imshow(diff, cmap='Reds')
        axes[1, 1].set_title('Prediction Error')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1])
        
        # Metrics
        mae = np.mean(diff)
        mse = np.mean(diff**2)
        axes[1, 2].axis('off')
        metrics_text = f"""
        Prediction Metrics:
        
        Mean Absolute Error: {mae:.4f}
        Mean Squared Error: {mse:.4f}
        
        Volume Shape: {predicted_volume.shape}
        Depth Range: [{predicted_volume.min():.3f}, {predicted_volume.max():.3f}]
        """
        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace')
    else:
        # Statistics for single prediction
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
        axes[1, 2].axis('off')
        
        stats_text = f"""
        Prediction Statistics:
        
        Volume Shape: {predicted_volume.shape}
        Value Range: [{predicted_volume.min():.3f}, {predicted_volume.max():.3f}]
        Mean Value: {predicted_volume.mean():.3f}
        Std Dev: {predicted_volume.std():.3f}
        
        Total Voxels: {predicted_volume.size:,}
        Non-zero Voxels: {(predicted_volume > 0.5).sum():,}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    return fig


def demo_inference():
    """Demonstrate inference"""
    print("\n" + "="*70)
    print("  3D CNN Volume Prediction - Inference Demo")
    print("="*70 + "\n")
    
    parser = argparse.ArgumentParser(description='3D Prediction Demo')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--depth', type=int, default=32, help='Volume depth')
    parser.add_argument('--num-samples', type=int, default=3,
                       help='Number of samples to process')
    
    args = parser.parse_args()
    
    # Initialize predictor
    print("Initializing predictor...")
    predictor = Volume3DPredictor(
        model_path=args.model,
        depth=args.depth,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Load test data
    print("Loading test dataset...")
    dataset = SyntheticPaired3DDataset(
        num_samples=args.num_samples,
        img_size=256,
        depth=args.depth,
        mode='test'
    )
    
    print(f"\nProcessing {args.num_samples} samples...\n")
    
    for idx in range(min(args.num_samples, len(dataset))):
        print(f"\n{'='*70}")
        print(f"  Sample {idx+1}/{min(args.num_samples, len(dataset))}")
        print(f"{'='*70}\n")
        
        image_2d, volume_gt = dataset[idx]
        
        # Convert to numpy for visualization
        image_2d_np = image_2d.squeeze().numpy()
        volume_gt_np = volume_gt.squeeze().numpy()
        
        print(f"Input shape: {image_2d_np.shape}")
        print(f"Ground truth shape: {volume_gt_np.shape}")
        
        # Predict
        print("Running inference...")
        volume_pred = predictor.predict(image_2d_np)
        print(f"Prediction shape: {volume_pred.shape}")
        
        # Visualize comparison
        print("Generating comparison visualization...")
        compare_predictions(
            image_2d_np,
            volume_pred,
            ground_truth_volume=volume_gt_np,
            save_path=f'demo_prediction_{idx+1}.png'
        )
        
        # Visualize full volume
        print("Generating full volume visualization...")
        visualize_volume_slices(
            volume_pred,
            title=f'Predicted 3D Volume - Sample {idx+1}',
            save_path=f'demo_volume_{idx+1}.png'
        )
        
        # Calculate metrics if ground truth available
        mae = np.mean(np.abs(volume_pred - volume_gt_np))
        mse = np.mean((volume_pred - volume_gt_np)**2)
        rmse = np.sqrt(mse)
        
        print(f"\nMetrics:")
        print(f"  MAE:  {mae:.6f}")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")


def demo_architecture():
    """Show model architecture info"""
    print("\n" + "="*70)
    print("  3D CNN Architecture")
    print("="*70 + "\n")
    
    model = Slice2Volume3DCNN(depth=32)
    
    print("Model Structure:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*70}")
    print(f"Total Parameters:    {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"{'='*70}\n")
    
    # Example forward pass
    print("Testing forward pass...")
    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        y = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print("\n✓ Model working correctly!")


if __name__ == '__main__':
    import sys
    
    # Check for command
    if len(sys.argv) > 1 and sys.argv[1] == '--architecture':
        demo_architecture()
    else:
        demo_inference()
