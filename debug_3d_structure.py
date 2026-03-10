"""
Debug script to test 3D visualization in the pipeline
"""
import numpy as np
from pathlib import Path
from PIL import Image
import json

# Create test image
test_img_path = Path('test_3d_debug.png')
test_image = np.random.normal(100, 20, (256, 256)).astype(np.uint8)
test_image_pil = Image.fromarray(test_image, mode='L')
test_image_pil.save(test_img_path)

print(f"Created test image: {test_img_path}")

# Import pipeline
from pipeline_segmentation_to_3d import SegmentationTo3D

# Initialize
pipeline = SegmentationTo3D(
    model_path='best_model.pth',
    model_3d_path='models_3d/final_3d_model.pth',
    device='cpu'
)

print("\n" + "="*70)
print("STEP 1: Segmentation")
print("="*70)
try:
    seg_result = pipeline.segment(str(test_img_path))
    print(f"✅ Segmentation successful")
    print(f"   Coverage: {seg_result['coverage_percent']:.2f}%")
    print(f"   Confidence: {seg_result['confidence']:.4f}")
except Exception as e:
    print(f"❌ Segmentation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*70)
print("STEP 2: 3D Reconstruction")
print("="*70)
try:
    recon_result = pipeline.reconstruct_3d(seg_result)
    print(f"✅ 3D reconstruction successful")
    print(f"   Volume shape: {recon_result['volume'].shape}")
    print(f"   Volume dtype: {recon_result['volume'].dtype}")
    print(f"   Volume min/max: {recon_result['volume'].min():.4f} / {recon_result['volume'].max():.4f}")
    print(f"   Depth map shape: {recon_result['depth_map'].shape}")
except Exception as e:
    print(f"❌ 3D reconstruction failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*70)
print("STEP 3: Generate Visualization")
print("="*70)
try:
    fig = pipeline.reconstructor.visualize_3d(
        recon_result,
        title="Debug Test",
        save_path=str(Path('debug_3d_viz.png'))
    )
    print(f"✅ Visualization generated")
    print(f"   Saved to: debug_3d_viz.png")
except Exception as e:
    print(f"❌ Visualization failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*70)
print("STEP 4: Generate Interactive HTML")
print("="*70)
try:
    html_output = Path('debug_3d_viewer.html')
    html_path = pipeline.reconstructor.create_interactive_html(
        recon_result,
        patient_info={
            'name': 'Debug Test',
            'id': 'DBG-001'
        },
        output_path=str(html_output)
    )
    print(f"✅ Interactive HTML generated")
    print(f"   Saved to: {html_output}")
    
    # Check file size
    file_size = html_output.stat().st_size / 1024
    print(f"   File size: {file_size:.1f} KB")
    
    # Check if Three.js code is present
    with open(html_output, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        if 'THREE' in content:
            print(f"   ✅ Three.js viewer code present")
        else:
            print(f"   ⚠️  Three.js code NOT found")
            
        if 'volumeData' in content:
            print(f"   ✅ Volume data embedded")
        else:
            print(f"   ⚠️  Volume data NOT found")
            
except Exception as e:
    print(f"❌ HTML generation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED - 3D Structure should be showing")
print("="*70)
print("\nDebug files created:")
print("  • debug_3d_viz.png - 6-panel visualization")
print("  • debug_3d_viewer.html - Interactive 3D viewer")
print("\nOpen debug_3d_viewer.html in a browser to test the 3D viewer")
