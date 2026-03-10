"""
Quick Start Guide - 3D Medical Image Reconstruction
====================================================
Run this script to:
1. Train a 3D volume predictor (5-10 minutes)
2. Test on sample images
3. Generate 3D reconstructions
4. Launch web interface

Usage:
    python quickstart_3d.py [--skip-training] [--device cuda]
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json
import numpy as np
from PIL import Image

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def create_sample_image():
    """Create a sample medical image for testing"""
    print("Creating sample medical image...")
    
    # Create synthetic brain-like image
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Add brain-like structures
    from scipy.ndimage import gaussian_filter
    
    # Ventricles (center)
    y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
    mask = (x**2 + y**2) <= (30**2)
    img[mask] = 200
    
    # Brain tissue
    mask = (x**2 + y**2) <= (80**2)
    img[mask] = 150
    
    # Add some noise and smoothing
    noise = np.random.randint(0, 20, (size, size))
    img = img + noise
    img = gaussian_filter(img.astype(float), sigma=2)
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Save
    sample_path = Path('sample_brain.jpg')
    Image.fromarray(img).save(sample_path)
    print(f"✓ Sample image saved: {sample_path}")
    
    return str(sample_path)

def train_3d_model():
    """Train 3D volume predictor"""
    print_header("STEP 1: Training 3D Volume Predictor")
    
    print("This will:")
    print("  1. Generate 500 synthetic 2D-3D image pairs")
    print("  2. Train a 3D CNN for 50 epochs")
    print("  3. Save best model to models_3d/")
    print("  4. Take ~5-10 minutes on CPU, ~1-2 minutes on GPU\n")
    
    input("Press Enter to start training...")
    
    cmd = [
        sys.executable, 'train_3d_predictor.py',
        '--num-samples', '500',
        '--epochs', '50',
        '--batch-size', '8',
        '--output-dir', './models_3d'
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print("⚠️  Training failed")
        return False
    
    print_header("✓ Training Complete!")
    return True

def find_best_model():
    """Find the best trained model"""
    models_dir = Path('models_3d')
    if not models_dir.exists():
        return None
    
    # Find best_3d_model_epochX.pth
    best_models = list(models_dir.glob('best_3d_model_epoch*.pth'))
    if best_models:
        return str(best_models[-1])
    
    return None

def test_pipeline(image_path, model_3d_path):
    """Test the complete pipeline"""
    print_header("STEP 2: Testing Complete Pipeline")
    
    print(f"Processing image: {image_path}")
    print(f"Using 3D model: {model_3d_path}\n")
    
    cmd = [
        sys.executable, 'pipeline_segmentation_to_3d.py',
        '--image', image_path,
        '--model', 'best_model.pth',
        '--model-3d', model_3d_path,
        '--patient-name', 'Sample Patient',
        '--patient-id', 'SAMPLE001',
        '--findings', 'Generated for demonstration',
        '--output', './reconstruction_output'
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print("⚠️  Pipeline processing failed")
        return False
    
    print_header("✓ Processing Complete!")
    return True

def show_results():
    """Show available results"""
    print_header("STEP 3: Generated Results")
    
    output_dir = Path('reconstruction_output')
    
    if not output_dir.exists():
        print("No results directory found")
        return
    
    print("Generated files:")
    for file in sorted(output_dir.glob('**/*')):
        if file.is_file():
            size_kb = file.stat().st_size / 1024
            print(f"  • {file.relative_to(output_dir)} ({size_kb:.1f} KB)")
    
    # Show report
    report_file = output_dir / 'report.json'
    if report_file.exists():
        print("\n" + "-"*70)
        print("Analysis Report:")
        print("-"*70)
        with open(report_file) as f:
            report = json.load(f)
        print(json.dumps(report, indent=2))

def launch_web_interface():
    """Launch web interface"""
    print_header("STEP 4: Launch Web Interface (Optional)")
    
    print("The web interface allows you to:")
    print("  • Upload medical images via drag-and-drop")
    print("  • Add patient information")
    print("  • View interactive 3D visualizations")
    print("  • Download analysis reports")
    print()
    print("To start: python web_interface_3d.py")
    print("Then open: http://localhost:5000")
    print()
    
    response = input("Launch web interface now? (y/n): ").lower()
    if response == 'y':
        try:
            subprocess.run([sys.executable, 'web_interface_3d.py'])
        except KeyboardInterrupt:
            print("\n✓ Web interface stopped")

def show_summary():
    """Show summary of what was created"""
    print_header("Summary")
    
    summary = """
🎯 You now have a complete 3D medical image reconstruction system!

📁 Key Files Created:
   • model_3d_prediction.py      - 3D CNN architecture
   • train_3d_predictor.py       - Training script
   • pipeline_segmentation_to_3d.py - End-to-end pipeline
   • web_interface_3d.py         - Web UI
   • reconstruction_3d.py        - 3D visualization utilities
   • README_3D_RECONSTRUCTION.md - Detailed documentation

📊 Models Available:
   • best_model.pth                    - 2D Segmentation (existing)
   • models_3d/best_3d_model_epochX.pth - 3D Volume Prediction (trained)

🚀 Next Steps:

   1. Train Model (if not done):
      python train_3d_predictor.py --epochs 50

   2. Process Images:
      python pipeline_segmentation_to_3d.py \\
         --image scan.jpg \\
         --model best_model.pth \\
         --model-3d models_3d/best_3d_model_epoch50.pth

   3. Use Web Interface:
      python web_interface_3d.py

   4. Advanced Usage:
      See README_3D_RECONSTRUCTION.md for complete documentation

💡 Tips:
   • Use GPU for faster training: --device cuda
   • Adjust depth for different image resolutions: --depth 16/32/64
   • Provide real paired data for better accuracy
   • Monitor training_history.json for convergence

📚 Learn More:
   • Check README_3D_RECONSTRUCTION.md for detailed docs
   • Review model_3d_prediction.py for architecture details
   • See examples in pipeline_segmentation_to_3d.py

Questions? Check the inline documentation in Python files!
    """
    
    print(summary)

def main():
    parser = argparse.ArgumentParser(description='3D Reconstruction Quick Start')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for training')
    parser.add_argument('--skip-web', action='store_true',
                       help='Skip web interface launch')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  🔬 3D Medical Image Reconstruction - Quick Start")
    print("="*70)
    
    # Check if model exists
    best_model = Path('best_model.pth')
    if not best_model.exists():
        print("\n⚠️  Warning: best_model.pth not found!")
        print("   The 2D segmentation model is required.")
        print("   Please ensure it exists in the current directory.")
        sys.exit(1)
    
    # Step 1: Training
    model_3d_path = None
    
    if not args.skip_training:
        success = train_3d_model()
        if not success:
            print("\n⚠️  Training failed, but you can still test with a new model...")
            model_3d_path = None
        else:
            model_3d_path = find_best_model()
    else:
        # Try to find existing model
        model_3d_path = find_best_model()
        if model_3d_path:
            print(f"✓ Found existing 3D model: {model_3d_path}")
        else:
            print("⚠️  No trained 3D model found. Train first with: python train_3d_predictor.py")
    
    if not model_3d_path:
        print("\n⚠️  Cannot proceed without a 3D model")
        print("   Please train first: python train_3d_predictor.py")
        sys.exit(1)
    
    # Step 2: Create sample image and test
    sample_image = create_sample_image()
    success = test_pipeline(sample_image, model_3d_path)
    
    if success:
        # Step 3: Show results
        show_results()
        
        # Step 4: Web interface (optional)
        if not args.skip_web:
            launch_web_interface()
    
    # Summary
    show_summary()

if __name__ == '__main__':
    main()
