"""
End-to-End Pipeline: 2D Segmentation → 3D Reconstruction
=========================================================
Integrates medical image segmentation with 3D reconstruction and visualization

Features:
- Load 2D MRI/X-ray images
- Run segmentation with trained model
- Generate 3D reconstruction
- Create interactive visualizations
- Export patient reports with findings

Usage:
    python pipeline_segmentation_to_3d.py --image input.jpg --model best_model.pth
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import json
import tempfile
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Import existing model and reconstruction modules
try:
    from main_implementation import AttentionUNet
    from reconstruction_3d import Reconstruction3D
    from model_3d_prediction import Volume3DPredictor
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure main_implementation.py, reconstruction_3d.py, and model_3d_prediction.py are in the same directory")


class SegmentationTo3D:
    """
    Complete pipeline from 2D segmentation to 3D reconstruction
    Uses 3D CNN for accurate volume prediction
    """
    
    def __init__(self, model_path: str, model_3d_path: str = None, device: str = 'cpu'):
        """
        Initialize pipeline
        
        Args:
            model_path: Path to trained Attention U-Net model (2D segmentation)
            model_3d_path: Path to trained 3D volume prediction model (optional)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Initialize 3D predictor
        self.predictor_3d = Volume3DPredictor(
            model_path=model_3d_path,
            depth=32,
            device=device
        )
        
        self.reconstructor = Reconstruction3D(device=device, method='cnn')
        
        print(f"[OK] Pipeline initialized")
        print(f"  2D Segmentation: Attention U-Net")
        print(f"  3D Prediction: 3D CNN ({model_3d_path if model_3d_path else 'untrained'})")
        print(f"  Device: {device}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model"""
        model = AttentionUNet(in_channels=1, num_classes=1)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        return model
    
    def _preprocess_image(self, image_path: str, size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess image
        
        Args:
            image_path: Path to image file
            size: Target size (will be resized to size×size)
            
        Returns:
            Tuple of (preprocessed tensor, original image array)
        """
        # Load image
        if isinstance(image_path, str):
            img = Image.open(image_path)
        else:
            img = image_path
        
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        
        original = np.array(img)
        
        # Resize
        img_resized = img.resize((size, size), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32)
        
        # Normalize to 0-1
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
        
        # Convert to tensor (1, 1, H, W)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(self.device)
        
        return img_tensor, original, img_array
    
    def segment(self, image_path: str, confidence_threshold: float = 0.5) -> Dict:
        """
        Run segmentation on image
        
        Args:
            image_path: Path to medical image
            confidence_threshold: Threshold for binary segmentation
            
        Returns:
            Dictionary with segmentation results
        """
        # Preprocess
        img_tensor, original, img_array = self._preprocess_image(image_path)
        
        # Inference
        with torch.no_grad():
            output = self.model(img_tensor)
            # Sigmoid for binary segmentation
            segmentation = torch.sigmoid(output).cpu().numpy()[0, 0]
        
        # Threshold
        segmentation_binary = (segmentation > confidence_threshold).astype(np.float32)
        
        # Calculate metrics
        area_pixels = np.sum(segmentation_binary)
        total_pixels = segmentation_binary.size
        coverage = (area_pixels / total_pixels) * 100
        
        return {
            'original_image': original,
            'preprocessed_image': img_array,
            'segmentation_prob': segmentation,
            'segmentation_binary': segmentation_binary,
            'confidence': np.mean(segmentation[segmentation_binary > 0]) if area_pixels > 0 else 0,
            'coverage_percent': coverage,
            'area_pixels': int(area_pixels)
        }
    
    def reconstruct_3d(self, segmentation_result: Dict, thickness: float = 10.0) -> Dict:
        """
        Create 3D reconstruction from segmentation using 3D CNN
        
        Args:
            segmentation_result: Output from segment()
            thickness: Slice thickness in mm
            
        Returns:
            3D reconstruction data
        """
        # Get original preprocessed image for 3D CNN prediction
        original_img = segmentation_result['preprocessed_image']
        
        # Predict 3D volume using trained CNN
        volume_3d = self.predictor_3d.predict(original_img)
        
        # Scale volume based on segmentation confidence
        seg_mask = segmentation_result['segmentation_binary']
        volume_3d = volume_3d * seg_mask[np.newaxis, :, :]  # Gate volume with segmentation
        
        reconstruction = {
            'volume': volume_3d,
            'depth_map': volume_3d.mean(axis=0),  # Average across depth
            'dimensions': volume_3d.shape,
            'thickness_mm': thickness,
            'method': '3D CNN Prediction',
            'confidence': float(segmentation_result['confidence'])
        }
        
        return reconstruction
    
    def process_complete(
        self,
        image_path: str,
        patient_info: Optional[Dict] = None,
        findings: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Complete pipeline: segment + reconstruct + visualize
        
        Args:
            image_path: Path to medical image
            patient_info: Patient metadata
            findings: Clinical findings text
            output_dir: Directory to save outputs
            
        Returns:
            Complete processing results
        """
        print(f"\n{'='*60}")
        print(f"Processing: {Path(image_path).name}")
        print(f"{'='*60}")
        
        # Step 1: Segmentation
        print("Step 1: Running segmentation...")
        seg_result = self.segment(image_path)
        print(f"  [OK] Segmentation complete")
        print(f"    Coverage: {seg_result['coverage_percent']:.2f}%")
        print(f"    Confidence: {seg_result['confidence']:.4f}")
        
        # Step 2: 3D Reconstruction
        print("Step 2: Creating 3D reconstruction...")
        recon_result = self.reconstruct_3d(seg_result)
        print(f"  [OK] Reconstruction complete")
        print(f"    Dimensions: {recon_result['dimensions']}")
        
        # Step 3: Visualization
        print("Step 3: Generating visualizations...")
        
        # Create output directory
        if output_dir is None:
            output_dir = Path(image_path).parent / "reconstruction_output"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save segmentation visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(seg_result['preprocessed_image'], cmap='gray')
        axes[0].imshow(seg_result['segmentation_prob'], alpha=0.5, cmap='hot')
        axes[0].set_title('Segmentation Probability')
        axes[0].axis('off')
        
        axes[1].imshow(seg_result['segmentation_binary'], cmap='Greens')
        axes[1].set_title('Binary Segmentation')
        axes[1].axis('off')
        
        axes[2].imshow(recon_result['depth_map'], cmap='viridis')
        axes[2].set_title('Estimated Depth Map')
        axes[2].axis('off')
        
        plt.tight_layout()
        seg_viz_path = output_dir / "segmentation_and_depth.png"
        plt.savefig(seg_viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: segmentation_and_depth.png")
        
        # Save 3D reconstruction visualization
        recon_viz_path = output_dir / "3d_reconstruction.png"
        self.reconstructor.visualize_3d(recon_result, save_path=str(recon_viz_path))
        print(f"  [OK] Saved: 3d_reconstruction.png")
        
        # Create interactive HTML
        html_path = output_dir / "3d_viewer.html"
        self.reconstructor.create_interactive_html(
            recon_result,
            patient_info=patient_info,
            findings=findings,
            output_path=str(html_path),
            segmentation_data={
                'coverage_percent': seg_result['coverage_percent'],
                'confidence': seg_result['confidence']
            }
        )
        print(f"  [OK] Saved: 3d_viewer.html")
        
        # Create comprehensive report
        print("Step 4: Creating report...")
        report = self._generate_report(
            seg_result,
            recon_result,
            patient_info,
            findings
        )
        
        report_path = output_dir / "report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        print(f"  [OK] Saved: report.json")
        
        print(f"\n{'='*60}")
        print(f"[OK] Processing complete!")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")
        
        return {
            'segmentation': seg_result,
            'reconstruction': recon_result,
            'report': report,
            'output_dir': str(output_dir),
            'files': {
                'segmentation_viz': str(seg_viz_path),
                'reconstruction_viz': str(recon_viz_path),
                '3d_viewer': str(html_path),
                'report': str(report_path)
            }
        }
    
    @staticmethod
    def _generate_report(
        seg_result: Dict,
        recon_result: Dict,
        patient_info: Optional[Dict] = None,
        findings: Optional[str] = None
    ) -> Dict:
        """Generate comprehensive report"""
        report = {
            'patient_info': patient_info or {},
            'clinical_findings': findings or 'Not specified',
            'segmentation': {
                'method': 'Attention U-Net',
                'coverage_percent': round(seg_result['coverage_percent'], 2),
                'confidence_score': round(float(seg_result['confidence']), 4),
                'area_pixels': seg_result['area_pixels']
            },
            'reconstruction': {
                'method': recon_result['method'],
                'depth_estimation': 'Edge-based',
                'dimensions': recon_result['dimensions'],
                'thickness_mm': recon_result['thickness_mm']
            },
            'timestamp': str(Path.cwd()),
            'status': 'SUCCESS'
        }
        return report


# CLI Interface
def main():
    parser = argparse.ArgumentParser(
        description='2D Medical Image Segmentation to 3D Reconstruction Pipeline'
    )
    parser.add_argument('--image', type=str, required=True, help='Path to medical image')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to trained 2D segmentation model')
    parser.add_argument('--model-3d', type=str, help='Path to trained 3D volume prediction model')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--patient-name', type=str, help='Patient name')
    parser.add_argument('--patient-id', type=str, help='Patient ID')
    parser.add_argument('--findings', type=str, help='Clinical findings')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SegmentationTo3D(args.model, model_3d_path=args.model_3d, device=args.device)
    
    # Prepare patient info
    patient_info = {}
    if args.patient_name:
        patient_info['Name'] = args.patient_name
    if args.patient_id:
        patient_info['ID'] = args.patient_id
    
    # Process image
    results = pipeline.process_complete(
        image_path=args.image,
        patient_info=patient_info if patient_info else None,
        findings=args.findings,
        output_dir=args.output
    )
    
    # Print summary
    print("\nReport Summary:")
    print(json.dumps(results['report'], indent=2, cls=NumpyEncoder))


if __name__ == '__main__':
    main()
