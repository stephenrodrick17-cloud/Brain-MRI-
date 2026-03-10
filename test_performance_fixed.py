"""
Performance Testing Script for 3D Medical Image Reconstruction
Measures end-to-end processing time with optimizations
"""

import time
import numpy as np
from pathlib import Path
import json
from PIL import Image

# Import pipeline
from pipeline_segmentation_to_3d import SegmentationTo3D

def create_test_image():
    """Create a synthetic test medical image (grayscale, 256x256)"""
    np.random.seed(42)
    
    # Create synthetic medical image (grayscale for medical imaging)
    img = np.random.normal(100, 20, (256, 256)).astype(np.uint8)
    
    # Add a synthetic structure (circle/sphere-like)
    y, x = np.ogrid[:256, :256]
    mask = (x - 128) ** 2 + (y - 128) ** 2 <= 50 ** 2
    img[mask] = np.clip(img[mask] + 80, 0, 255)
    
    return img

def test_performance():
    """Test and measure reconstruction performance"""
    
    print("\n" + "=" * 70)
    print("📊 3D RECONSTRUCTION PERFORMANCE TEST")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path('performance_test_output')
    output_dir.mkdir(exist_ok=True)
    
    # Create test image
    print("\n[Step 1] Creating test synthetic medical image...")
    test_img_path = output_dir / 'test_image.png'
    test_image = create_test_image()
    # Convert to PIL Image and save
    pil_img = Image.fromarray(test_image, mode='L')
    pil_img.save(test_img_path)
    print(f"  ✓ Created test image: {test_img_path}")
    print(f"    Shape: {test_image.shape}, Dtype: {test_image.dtype}")
    
    # Initialize pipeline
    print("\n[Step 2] Initializing 3D reconstruction pipeline...")
    start = time.time()
    pipeline = SegmentationTo3D(
        model_path='best_model.pth',
        model_3d_path='models_3d/final_3d_model.pth',
        device='cpu'
    )
    init_time = time.time() - start
    print(f"  ✓ Pipeline initialized in {init_time:.2f}s")
    
    # Run reconstruction
    print("\n[Step 3] Running 3D reconstruction pipeline...")
    print("  Processing image...")
    
    timing = {}
    
    try:
        # Segmentation
        print("    - 2D Segmentation (Attention U-Net)...", end='', flush=True)
        start = time.time()
        seg_result = pipeline.segment(str(test_img_path))
        timing['segmentation'] = time.time() - start
        print(f" {timing['segmentation']:.2f}s")
        
        # 3D Reconstruction
        print("    - 3D CNN Volume Prediction...", end='', flush=True)
        start = time.time()
        recon_result = pipeline.reconstruct_3d(seg_result)
        timing['3d_reconstruction'] = time.time() - start
        print(f" {timing['3d_reconstruction']:.2f}s")
        
        # Visualization
        print("    - Generating 6-Panel Visualization...", end='', flush=True)
        start = time.time()
        fig = pipeline.reconstructor.visualize_3d(
            recon_result,
            title="Performance Test Result",
            save_path=str(output_dir / 'visualization_perf_test.png')
        )
        timing['visualization'] = time.time() - start
        print(f" {timing['visualization']:.2f}s")
        
        # HTML Report
        print("    - Generating Professional HTML Report...", end='', flush=True)
        start = time.time()
        html_path = pipeline.reconstructor.create_interactive_html(
            recon_result,
            patient_info={
                'name': 'Test Patient',
                'id': 'PERF-TEST-001',
                'age': '45',
                'gender': 'M'
            },
            findings='Performance test reconstruction - synthetic data',
            output_path=str(output_dir / 'viewer_perf_test.html'),
            segmentation_data={
                'coverage_percent': seg_result['coverage_percent'],
                'confidence': float(seg_result['confidence'])
            }
        )
        timing['html_generation'] = time.time() - start
        print(f" {timing['html_generation']:.2f}s")
        
        # Total time
        total_time = sum(timing.values())
        
        # Results summary
        print("\n" + "=" * 70)
        print("📈 PERFORMANCE RESULTS (with optimizations)")
        print("=" * 70)
        print(f"\nTiming Breakdown:")
        print(f"  • 2D Segmentation:        {timing['segmentation']:>8.2f}s")
        print(f"  • 3D Reconstruction:      {timing['3d_reconstruction']:>8.2f}s")
        print(f"  • Visualization (6-panel):{timing['visualization']:>8.2f}s")
        print(f"  • HTML Report Generation: {timing['html_generation']:>8.2f}s")
        print(f"  " + "-" * 40)
        print(f"  • TOTAL TIME:             {total_time:>8.2f}s")
        
        print(f"\nOptimizations Applied:")
        print(f"  ✓ Model inference: float32 dtype + faster CPU ops")
        print(f"  ✓ Visualization: DPI 100 (reduced from 150)")
        print(f"  ✓ Histograms: 30 bins (reduced from 50)")
        print(f"  ✓ Memory: plt.close() for cleanup")
        print(f"  ✓ Server: Debug mode disabled, no watchdog restarts")
        
        print(f"\nOutput Files:")
        print(f"  ✓ Visualization: {output_dir / 'visualization_perf_test.png'}")
        print(f"  ✓ Interactive 3D: {output_dir / 'viewer_perf_test.html'}")
        
        print("\n" + "=" * 70)
        
        # Save results to JSON
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_type': 'Performance Benchmark',
            'timing': {k: float(v) for k, v in timing.items()},
            'total_time': float(total_time),
            'average_time_per_step': float(total_time / len(timing)),
            'optimizations': [
                'float32 dtype conversion',
                'reduced DPI visualization',
                'reduced histogram bins',
                'memory cleanup with plt.close()',
                'debug mode disabled'
            ]
        }
        
        results_path = output_dir / 'performance_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
        print("=" * 70 + "\n")
        
        return timing, total_time
        
    except Exception as e:
        print(f"\n❌ Error in processing: {e}")
        raise

if __name__ == '__main__':
    try:
        timing, total = test_performance()
        print(f"\n✅ Performance test completed successfully!")
        print(f"   Total processing time: {total:.2f} seconds")
    except Exception as e:
        print(f"\n❌ Error during performance test: {e}")
        import traceback
        traceback.print_exc()
