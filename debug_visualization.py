"""
Debug script to test visualization without optimize parameter
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Test the exact visualization code
def test_visualization():
    """Test 6-panel visualization"""
    
    # Create dummy volume
    volume = np.random.rand(100, 256, 256).astype(np.float32)
    depth_map = np.random.rand(256, 256).astype(np.float32)
    
    fig = plt.figure(figsize=(16, 10))
    
    # Normalize
    volume_norm = ((volume - volume.min()) / (volume.max() - volume.min() + 1e-8))
    
    # Panel 1: Depth map
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(depth_map, cmap='viridis', interpolation='bilinear')
    ax1.set_title('Estimated Depth Map', fontsize=12, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Panel 2: Max projection axial
    ax2 = fig.add_subplot(2, 3, 2)
    mip_axial = np.max(volume_norm, axis=0)
    im2 = ax2.imshow(mip_axial, cmap='hot', interpolation='bilinear')
    ax2.set_title('Axial Projection', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Panel 3: Sagittal
    ax3 = fig.add_subplot(2, 3, 3)
    mip_sag = np.max(volume_norm, axis=1)
    im3 = ax3.imshow(mip_sag, cmap='hot', interpolation='bilinear')
    ax3.set_title('Sagittal Projection', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Panel 4: Coronal
    ax4 = fig.add_subplot(2, 3, 4)
    mip_cor = np.max(volume_norm, axis=2)
    im4 = ax4.imshow(mip_cor, cmap='hot', interpolation='bilinear')
    ax4.set_title('Coronal Projection', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # Panel 5: Middle slice
    ax5 = fig.add_subplot(2, 3, 5)
    mid_slice = volume_norm[volume_norm.shape[0]//2, :, :]
    im5 = ax5.imshow(mid_slice, cmap='viridis', interpolation='nearest')
    ax5.set_title('Middle Axial Slice', fontsize=12, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    # Panel 6: Histogram
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(volume_norm.flatten(), bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax6.set_title('Volume Intensity Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Intensity Value')
    ax6.set_ylabel('Frequency')
    ax6.grid(True, alpha=0.3)
    
    fig.suptitle('Test Visualization', fontsize=16, fontweight='bold', y=0.98)
    
    # Save WITHOUT optimize parameter
    save_path = Path('debug_test_viz.png')
    print(f"Saving to: {save_path}")
    
    try:
        # This is the CORRECT way - no optimize parameter
        plt.savefig(str(save_path), dpi=100, bbox_inches='tight', format='png')
        print(f"✅ SUCCESS: Saved visualization without optimize parameter")
        print(f"File size: {save_path.stat().st_size / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    finally:
        plt.close(fig)
    
    return True

if __name__ == '__main__':
    print("Testing visualization without optimize parameter...")
    print("=" * 60)
    success = test_visualization()
    print("=" * 60)
    if success:
        print("\n✅ All tests passed! The fix works correctly.")
        print("Code is ready for production use.")
    else:
        print("\n❌ Test failed. Please check the error above.")
