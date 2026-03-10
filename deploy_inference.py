"""
Medical Image Segmentation - Inference & Deployment Script
============================================================
Production-ready inference pipeline for clinical deployment

Features:
- Batch processing
- DICOM support
- Quality control checks
- Confidence scoring
- Visualization export
- API endpoint ready

Usage:
    python deploy_inference.py --input /path/to/images --output /path/to/results
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from flask import Flask, request, jsonify, render_template_string, send_file
from PIL import Image
import io
import tempfile
import os
import base64
import uuid


class InferencePipeline:
    """
    Production-ready inference pipeline with quality control and monitoring
    """
    
    def __init__(self, model_path: str, device: str = 'cpu', 
                 confidence_threshold: float = 0.5):
        """
        Initialize inference pipeline
        
        Args:
            model_path: Path to trained model weights
            device: 'cpu' or 'cuda'
            confidence_threshold: Threshold for binary segmentation
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Load model architecture (copy from main implementation)
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Device: {self.device}")
        print(f"✓ Confidence threshold: {self.confidence_threshold}")
    
    def _load_model(self, model_path: str):
        """Load trained model"""
        # Import model architecture
        try:
            # Prefer local implementation from main_implementation if available
            from main_implementation import AttentionUNet
        except Exception:
            try:
                from attention_unet import AttentionUNet  # Fallback module name
            except Exception:
                # Fallback minimal AttentionUNet implementation to avoid import errors.
                # Replace with the real implementation or install the module for production use.
                class AttentionUNet(nn.Module):
                    def __init__(self, in_channels=1, num_classes=1):
                        super().__init__()
                        self.encoder = nn.Sequential(
                            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(16, 32, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True)
                        )
                        self.decoder = nn.Sequential(
                            nn.Conv2d(32, 16, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(16, num_classes, kernel_size=1)
                        )

                    def forward(self, x):
                        x = self.encoder(x)
                        x = self.decoder(x)
                        return x
        
        model = AttentionUNet(in_channels=1, num_classes=1)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess single image for inference
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Ensure 2D
        if len(image.shape) == 3:
            image = image[:, :, 0]
        
        # Normalize (Z-score)
        mean = np.mean(image)
        std = np.std(image)
        normalized = (image - mean) / (std + 1e-8)
        
        # Clip outliers
        normalized = np.clip(normalized, -3, 3)
        
        # Resize to model input size
        from scipy.ndimage import zoom
        target_size = (256, 256)
        zoom_factors = (target_size[0] / image.shape[0], 
                       target_size[1] / image.shape[1])
        resized = zoom(normalized, zoom_factors, order=3)
        
        # Convert to tensor
        tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0)
        
        return tensor
    
    def postprocess_output(self, output: torch.Tensor, 
                          original_shape: tuple) -> np.ndarray:
        """
        Postprocess model output to original image size
        
        Args:
            output: Model output tensor
            original_shape: Original image dimensions
            
        Returns:
            Segmentation mask in original size
        """
        # Apply sigmoid and threshold
        prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
        binary_mask = (prob_map > self.confidence_threshold).astype(np.uint8)
        
        # Resize to original dimensions
        from scipy.ndimage import zoom
        zoom_factors = (original_shape[0] / binary_mask.shape[0],
                       original_shape[1] / binary_mask.shape[1])
        resized_mask = zoom(binary_mask, zoom_factors, order=0)
        resized_prob = zoom(prob_map, zoom_factors, order=1)
        
        return resized_mask, resized_prob
    
    def calculate_uncertainty(self, output: torch.Tensor, 
                             num_samples: int = 10) -> float:
        """
        Estimate prediction uncertainty using Monte Carlo Dropout
        
        Args:
            output: Model output
            num_samples: Number of stochastic forward passes
            
        Returns:
            Uncertainty score (higher = more uncertain)
        """
        # output here is expected to be the input tensor for the model
        input_tensor = output

        # If the model contains Dropout modules, use MC Dropout
        has_dropout = any(isinstance(m, nn.Dropout) for m in self.model.modules())

        if has_dropout:
            self.model.train()
            preds = []
            with torch.no_grad():
                for _ in range(num_samples):
                    p = torch.sigmoid(self.model(input_tensor)).cpu().numpy()
                    preds.append(p)
            self.model.eval()
            preds = np.array(preds)
            uncertainty = float(np.mean(np.var(preds, axis=0)))
            return uncertainty

        # Fallback: estimate uncertainty via predictive entropy from a single pass
        self.model.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.model(input_tensor)).squeeze().cpu().numpy()
        eps = 1e-6
        p = np.clip(p, eps, 1 - eps)
        entropy = - (p * np.log(p) + (1 - p) * np.log(1 - p))
        return float(np.mean(entropy))
    
    def quality_control(self, image: np.ndarray, mask: np.ndarray) -> dict:
        """
        Perform quality control checks on segmentation
        
        Args:
            image: Input image
            mask: Predicted segmentation mask
            
        Returns:
            Dictionary of QC metrics and flags
        """
        qc_results = {
            'passed': True,
            'warnings': [],
            'metrics': {}
        }
        
        # Check 1: Segmentation size reasonability
        mask_ratio = np.sum(mask) / mask.size
        qc_results['metrics']['mask_ratio'] = float(mask_ratio)
        
        if mask_ratio < 0.01:
            qc_results['warnings'].append('Very small segmentation (<1% of image)')
            qc_results['passed'] = False
        elif mask_ratio > 0.8:
            qc_results['warnings'].append('Very large segmentation (>80% of image)')
            qc_results['passed'] = False
        
        # Check 2: Connected components
        from scipy.ndimage import label
        labeled_mask, num_components = label(mask)
        qc_results['metrics']['num_components'] = int(num_components)
        
        if num_components > 10:
            qc_results['warnings'].append(f'Many disconnected regions ({num_components})')
        
        # Check 3: Image quality
        if np.std(image) < 0.01:
            qc_results['warnings'].append('Very low image contrast')
            qc_results['passed'] = False
        
        return qc_results
    
    def predict(self, image: np.ndarray, 
                uncertainty_estimation: bool = True) -> dict:
        """
        Complete prediction pipeline with QC
        
        Args:
            image: Input medical image
            uncertainty_estimation: Whether to estimate uncertainty
            
        Returns:
            Dictionary containing:
                - segmentation_mask: Binary mask
                - probability_map: Continuous probability map
                - uncertainty: Uncertainty score
                - qc_results: Quality control results
                - metadata: Processing metadata
        """
        start_time = datetime.now()
        original_shape = image.shape
        
        # Preprocess
        input_tensor = self.preprocess_image(image).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Postprocess
        mask, prob_map = self.postprocess_output(output, original_shape)
        
        # Uncertainty estimation
        uncertainty = None
        if uncertainty_estimation:
            uncertainty = self.calculate_uncertainty(input_tensor)
        
        # Quality control
        qc_results = self.quality_control(image, mask)
        
        # Compile results
        results = {
            'segmentation_mask': mask,
            'probability_map': prob_map,
            'uncertainty': uncertainty,
            'qc_results': qc_results,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                'model_device': self.device,
                'confidence_threshold': self.confidence_threshold,
                'input_shape': original_shape,
                'output_shape': mask.shape
            }
        }
        
        return results
    
    def visualize_results(self, image: np.ndarray, results: dict, 
                         save_path: str = None):
        """
        Create comprehensive visualization of results
        
        Args:
            image: Original image
            results: Prediction results dictionary
            save_path: Path to save visualization (optional)
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Probability map
        im1 = axes[0, 1].imshow(results['probability_map'], cmap='jet', vmin=0, vmax=1)
        axes[0, 1].set_title('Probability Map', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        # Binary mask
        axes[0, 2].imshow(results['segmentation_mask'], cmap='gray')
        axes[0, 2].set_title('Binary Segmentation', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Overlay on original
        axes[1, 0].imshow(image, cmap='gray')
        axes[1, 0].imshow(results['segmentation_mask'], cmap='Reds', alpha=0.4)
        axes[1, 0].set_title('Overlay', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Uncertainty visualization: if probability map available, show uncertainty heatmap and histogram
        prob_map = results.get('probability_map', None)
        if prob_map is not None:
            try:
                # Ensure float in [0,1]
                p = prob_map.astype(np.float32)
                p = np.clip(p, 1e-6, 1 - 1e-6)

                # Uncertainty map: predictive entropy per-pixel
                entropy = - (p * np.log(p) + (1 - p) * np.log(1 - p))

                im2 = axes[1, 1].imshow(entropy, cmap='magma')
                axes[1, 1].set_title('Uncertainty (predictive entropy)', fontsize=14, fontweight='bold')
                axes[1, 1].axis('off')
                plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

                # Histogram of uncertainty values
                axes[1, 2].hist(entropy.ravel(), bins=50, color='coral')
                axes[1, 2].set_title('Uncertainty distribution', fontsize=12, fontweight='bold')
                axes[1, 2].set_xlabel('Entropy')
                axes[1, 2].set_ylabel('Frequency')
            except Exception:
                axes[1, 1].text(0.5, 0.5, 'Uncertainty\nNot Available', ha='center', va='center', fontsize=14)
                axes[1, 1].axis('off')
                axes[1, 2].axis('off')
        else:
            if results.get('uncertainty', None) is not None:
                axes[1, 1].text(0.5, 0.5, f"Uncertainty: {results['uncertainty']:.6f}", ha='center', va='center', fontsize=14)
                axes[1, 1].axis('off')
            else:
                axes[1, 1].text(0.5, 0.5, 'Uncertainty\nNot Estimated', ha='center', va='center', fontsize=14)
                axes[1, 1].axis('off')
            axes[1, 2].axis('off')
        
        # QC metrics
        qc_text = f"Quality Control\n{'='*30}\n"
        qc_text += f"Status: {'PASSED' if results['qc_results']['passed'] else 'FAILED'}\n\n"
        qc_text += f"Metrics:\n"
        for key, value in results['qc_results']['metrics'].items():
            qc_text += f"  {key}: {value:.3f}\n"
        
        if results['qc_results']['warnings']:
            qc_text += f"\nWarnings:\n"
            for warning in results['qc_results']['warnings']:
                qc_text += f"  • {warning}\n"
        
        axes[1, 2].text(0.1, 0.9, qc_text, fontsize=10, verticalalignment='top',
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 2].axis('off')
        
        plt.suptitle(f"Medical Image Segmentation Results\nProcessing Time: {results['metadata']['processing_time_ms']:.1f}ms",
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
        plt.show()
    
    def batch_predict(self, image_paths: list, output_dir: str):
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of paths to input images
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_summary = []
        
        print(f"\nProcessing {len(image_paths)} images...")
        print("="*70)
        
        for idx, img_path in enumerate(image_paths):
            print(f"\n[{idx+1}/{len(image_paths)}] Processing: {img_path}")
            
            try:
                # Load image
                image = self._load_image(img_path)
                
                # Predict
                results = self.predict(image, uncertainty_estimation=True)
                
                # Save results
                img_name = Path(img_path).stem
                
                # Save mask
                mask_path = output_dir / f"{img_name}_mask.npy"
                np.save(mask_path, results['segmentation_mask'])
                
                # Save probability map
                prob_path = output_dir / f"{img_name}_prob.npy"
                np.save(prob_path, results['probability_map'])
                
                # Save visualization
                viz_path = output_dir / f"{img_name}_visualization.png"
                self.visualize_results(image, results, save_path=str(viz_path))
                
                # Save metadata
                metadata_path = output_dir / f"{img_name}_metadata.json"
                with open(metadata_path, 'w') as f:
                    # Convert numpy types to native Python types
                    metadata_serializable = {
                        'qc_results': {
                            'passed': results['qc_results']['passed'],
                            'warnings': results['qc_results']['warnings'],
                            'metrics': {k: float(v) for k, v in results['qc_results']['metrics'].items()}
                        },
                        'uncertainty': float(results['uncertainty']) if results['uncertainty'] else None,
                        'metadata': results['metadata']
                    }
                    json.dump(metadata_serializable, f, indent=4)
                
                # Summary
                results_summary.append({
                    'image': img_name,
                    'status': 'success',
                    'qc_passed': results['qc_results']['passed'],
                    'processing_time_ms': results['metadata']['processing_time_ms']
                })
                
                print(f"  ✓ Completed in {results['metadata']['processing_time_ms']:.1f}ms")
                print(f"  QC Status: {'PASSED' if results['qc_results']['passed'] else 'FAILED'}")
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                results_summary.append({
                    'image': Path(img_path).stem,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Save batch summary
        summary_path = output_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'total_images': len(image_paths),
                'successful': sum(1 for r in results_summary if r['status'] == 'success'),
                'failed': sum(1 for r in results_summary if r['status'] == 'failed'),
                'results': results_summary
            }, f, indent=4)
        
        print("\n" + "="*70)
        print("BATCH PROCESSING COMPLETED")
        print(f"✓ Total images: {len(image_paths)}")
        print(f"✓ Successful: {sum(1 for r in results_summary if r['status'] == 'success')}")
        print(f"✓ Failed: {sum(1 for r in results_summary if r['status'] == 'failed')}")
        print(f"✓ Results saved to: {output_dir}")
        print("="*70)
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load image from file (supports multiple formats)"""
        path = Path(path)
        
        if path.suffix in ['.npy']:
            return np.load(path)
        elif path.suffix in ['.png', '.jpg', '.jpeg']:
            from PIL import Image
            img = Image.open(path).convert('L')
            return np.array(img)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")


def create_app(model_path: str = 'best_model.pth', device: str = 'cpu', threshold: float = 0.5):
    """Create a small Flask app for uploading images and returning inference reports."""
    # Use non-interactive backend for server-side plotting
    import matplotlib
    matplotlib.use('Agg')

    app = Flask(__name__)
    pipeline = InferencePipeline(model_path=model_path, device=device, confidence_threshold=threshold)

    UPLOAD_HTML = '''
    <!doctype html>
    <title>Medical Image Segmentation - Upload</title>
    <h2>Upload X-ray / MRI Image (PNG / JPG / NPY)</h2>
    <form method=post enctype=multipart/form-data action="/predict">
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

    @app.route('/', methods=['GET'])
    def index():
        return render_template_string(UPLOAD_HTML)

    def _load_file_to_array(file_storage):
        # Try NPY / image / DICOM / NIfTI
        filename = file_storage.filename
        name, ext = os.path.splitext(filename.lower())
        file_bytes = file_storage.read()
        if ext == '.npy':
            arr = np.load(io.BytesIO(file_bytes))
            return arr
        elif ext in ('.png', '.jpg', '.jpeg'):
            img = Image.open(io.BytesIO(file_bytes)).convert('L')
            return np.array(img)
        elif ext in ('.dcm',):
            # Handle DICOM files
            try:
                import pydicom
            except Exception:
                raise RuntimeError('pydicom is required to read DICOM files; please install it in the venv')

            ds = pydicom.dcmread(io.BytesIO(file_bytes))

            # Multi-frame handling: try PixelData
            if hasattr(ds, 'pixel_array'):
                arr = ds.pixel_array
                # If multi-frame, keep as (Z,H,W) or pick first frame for 3D->2D
                # Do not discard frames here; caller will handle volumes

                # Apply rescale if present
                intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
                slope = float(getattr(ds, 'RescaleSlope', 1.0))
                arr = arr.astype(np.float32) * slope + intercept

                # Return array, modality, and window/level if available
                modality = getattr(ds, 'Modality', '').upper() if hasattr(ds, 'Modality') else ''
                wc = None
                ww = None
                if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                    try:
                        wc_val = ds.WindowCenter
                        ww_val = ds.WindowWidth
                        wc = float(wc_val[0]) if isinstance(wc_val, (list, tuple)) else float(wc_val)
                        ww = float(ww_val[0]) if isinstance(ww_val, (list, tuple)) else float(ww_val)
                    except Exception:
                        wc = None
                        ww = None

                return (arr, modality, wc, ww)
            else:
                raise RuntimeError('DICOM file does not contain pixel data')
        elif ext in ('.nii', '.nii.gz'):
            try:
                import nibabel as nib
            except Exception:
                raise RuntimeError('nibabel is required to read NIfTI files; please install it in the venv')
            img = nib.load(io.BytesIO(file_bytes))
            data = img.get_fdata()
            # Return 3D numpy array
            return data
        else:
            img = Image.open(io.BytesIO(file_bytes)).convert('L')
            return np.array(img)

    @app.route('/predict', methods=['POST'])
    def predict_route():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        try:
            loaded = _load_file_to_array(file)

            # If DICOM returned (arr, modality, wc, ww)
            modality = 'UNKNOWN'
            dicom_wc = None
            dicom_ww = None
            if isinstance(loaded, tuple):
                if len(loaded) == 4:
                    image_arr, modality, dicom_wc, dicom_ww = loaded
                else:
                    image_arr, modality = loaded
            else:
                image_arr = loaded

            # Create unique job id and directory
            job_id = uuid.uuid4().hex
            job_dir = os.path.join(tempfile.gettempdir(), f"med_infer_{job_id}")
            os.makedirs(job_dir, exist_ok=True)

            # If image_arr is a 3D volume (e.g., NIfTI or multi-frame DICOM)
            if isinstance(image_arr, np.ndarray) and image_arr.ndim == 3:
                z = image_arr.shape[0]
                masks = []
                probs = []
                for i in range(z):
                    slice_img = image_arr[i].astype(np.float32)
                    # Normalize slice for inference
                    if slice_img.max() <= 1.0:
                        slice_disp = (slice_img * 255).astype(np.uint8)
                    else:
                        slice_disp = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8) * 255).astype(np.uint8)

                    res = pipeline.predict(slice_disp.astype(np.float32), uncertainty_estimation=False)
                    masks.append(res['segmentation_mask'])
                    probs.append(res['probability_map'])

                masks = np.stack(masks, axis=0)
                probs = np.stack(probs, axis=0)

                # Save outputs
                mask_path = os.path.join(job_dir, 'mask.npy')
                prob_path = os.path.join(job_dir, 'prob.npy')
                np.save(mask_path, masks)
                np.save(prob_path, probs)
                # Save bundled NPZ (mask + probability)
                bundle_path = os.path.join(job_dir, 'bundle.npz')
                np.savez_compressed(bundle_path, mask=masks, probability=probs)

                # Save middle-slice visualization
                mid = z // 2
                pipeline.visualize_results((image_arr[mid]).astype(np.float32), {'segmentation_mask': masks[mid], 'probability_map': probs[mid], 'uncertainty': None, 'qc_results': {'passed': True, 'warnings': [], 'metrics': {}}, 'metadata': {'processing_time_ms': 0}}, save_path=os.path.join(job_dir, 'viz.png'))

                # Estimate uncertainty from probability maps (mean Bernoulli variance)
                try:
                    uncertainty = float(np.mean(probs * (1.0 - probs)))
                except Exception:
                    uncertainty = None

                # Save metadata
                metadata = {
                    'job_id': job_id,
                    'slices': int(z),
                    'modality': modality,
                    'uncertainty': uncertainty,
                    'files': {
                        'mask': 'mask.npy',
                        'probability_map': 'prob.npy',
                        'visualization': 'viz.png',
                        'bundle': 'bundle.npz'
                    }
                }
                with open(os.path.join(job_dir, 'metadata.json'), 'w') as mf:
                    json.dump(metadata, mf, indent=2)

                return jsonify({
                    'job_id': job_id,
                    'report_url': f"/report/{job_id}",
                    'files': {
                        'mask': f"/download/{job_id}/mask.npy",
                        'probability_map': f"/download/{job_id}/prob.npy",
                        'visualization': f"/download/{job_id}/viz.png",
                        'bundle': f"/download/{job_id}/bundle.npz"
                    },
                    'metadata': {'slices': z, 'modality': modality}
                })

            # Single-slice preprocessing (use DICOM window if available for CT)
            if modality == 'CT':
                if dicom_wc is not None and dicom_ww is not None:
                    wc, ww = dicom_wc, dicom_ww
                else:
                    wc, ww = 40, 400
                img = np.clip(image_arr, wc - ww/2, wc + ww/2)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img = (img * 255).astype(np.uint8)
            elif modality in ('MR', 'MRI'):
                img = image_arr.astype(np.float32)
                img = (img - np.mean(img)) / (np.std(img) + 1e-8)
                img = np.clip(img, -3, 3)
                img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
            else:
                img = image_arr.astype(np.float32)
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

            # Run inference
            results = pipeline.predict(img.astype(np.float32), uncertainty_estimation=True)

            # Save outputs for download and report
            mask_path = os.path.join(job_dir, 'mask.npy')
            prob_path = os.path.join(job_dir, 'prob.npy')
            viz_path = os.path.join(job_dir, 'viz.png')
            np.save(mask_path, results['segmentation_mask'])
            np.save(prob_path, results['probability_map'])
            # Save bundled NPZ for convenient download
            bundle_path = os.path.join(job_dir, 'bundle.npz')
            np.savez_compressed(bundle_path, mask=results['segmentation_mask'], probability=results['probability_map'])
            pipeline.visualize_results(img, results, save_path=viz_path)

            # Save metadata for this job (include uncertainty)
            # Embed the computed uncertainty into the saved metadata for easy reporting
            results['metadata']['uncertainty'] = results.get('uncertainty', None)
            metadata = {
                'job_id': job_id,
                'modality': modality,
                'qc_results': results['qc_results'],
                'files': {
                    'mask': 'mask.npy',
                    'probability_map': 'prob.npy',
                    'visualization': 'viz.png',
                    'bundle': 'bundle.npz'
                },
                'metadata': results['metadata'],
                'uncertainty': results.get('uncertainty', None)
            }
            with open(os.path.join(job_dir, 'metadata.json'), 'w') as mf:
                json.dump(metadata, mf, indent=2)

            with open(viz_path, 'rb') as vf:
                viz_b64 = base64.b64encode(vf.read()).decode('utf-8')

            interp = {
                'modality': modality,
                'qc_passed': results['qc_results']['passed'],
                'warnings': results['qc_results']['warnings'],
                'mask_ratio': results['qc_results']['metrics'].get('mask_ratio', None),
                'uncertainty': results['uncertainty']
            }

            return jsonify({
                'job_id': job_id,
                'report_url': f"/report/{job_id}",
                'interpretation': interp,
                'metadata': results['metadata'],
                'visualization_base64': viz_b64,
                'files': {
                    'mask': f"/download/{job_id}/mask.npy",
                    'probability_map': f"/download/{job_id}/prob.npy",
                    'visualization': f"/download/{job_id}/viz.png",
                    'bundle': f"/download/{job_id}/bundle.npz"
                }
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/result_image')
    def result_image():
        # Return latest file in temp dirs (simple helper)
        # In production, you'd implement secure storage and explicit links
        tmp_parent = tempfile.gettempdir()
        candidates = []
        for root, dirs, files in os.walk(tmp_parent):
            for f in files:
                if f.endswith('_viz.png'):
                    candidates.append(os.path.join(root, f))
        if not candidates:
            return 'No result image available', 404
        latest = max(candidates, key=os.path.getctime)
        return send_file(latest, mimetype='image/png')

    @app.route('/report/<job_id>')
    def report(job_id):
        job_dir = os.path.join(tempfile.gettempdir(), f"med_infer_{job_id}")
        meta_path = os.path.join(job_dir, 'metadata.json')
        if not os.path.exists(meta_path):
            return f"Report {job_id} not found", 404

        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        viz_path = os.path.join(job_dir, metadata['files'].get('visualization', 'viz.png'))
        # Read viz as base64 to embed
        viz_b64 = None
        if os.path.exists(viz_path):
            with open(viz_path, 'rb') as vf:
                viz_b64 = base64.b64encode(vf.read()).decode('utf-8')

                # Build a polished report and compute missing fields if needed
                REPORT_HTML = '''
                <!doctype html>
                <html>
                <head>
                    <meta charset="utf-8" />
                    <title>Inference Report - {{job_id}}</title>
                    <style>
                        :root { --accent: #0366d6; --muted:#666; }
                        body { font-family: Inter, Arial, Helvetica, sans-serif; margin: 20px; color: #222; background:#fbfdff }
                        .container { max-inline-size:1100px; margin:0 auto }
                        .top { display:flex; align-items:center; justify-content:space-between }
                        .card { background:#fff; border:1px solid #e6eef6; padding:14px; border-radius:10px; box-shadow:0 6px 18px rgba(15,23,42,0.04) }
                        .metrics { display:flex; gap:12px; margin-block-start:12px }
                        .metric { flex:1; padding:12px; text-align:center }
                        .metric h2 { margin:6px 0; font-size:20px }
                        .viz { inline-size:100%; border-radius:6px; border:1px solid #e9eef6 }
                        .downloads a { display:inline-block; margin:6px 8px 6px 0; padding:8px 12px; background:var(--accent); color:#fff; border-radius:6px; text-decoration:none }
                        .qc-pass { color: #0a7a3f; font-weight:700 }
                        .qc-fail { color: #c42b2b; font-weight:700 }
                        table.info { inline-size:100%; border-collapse:collapse; margin-block-start:12px }
                        table.info td { padding:8px 6px; border-block-end:1px solid #f1f5f9 }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="top">
                            <div>
                                <h1>Inference Report</h1>
                                <div style="color:var(--muted)">Job: <strong>{{job_id}}</strong></div>
                            </div>
                            <div style="text-align:end">
                                <div style="font-size:0.9rem;color:var(--muted)">Generated</div>
                                <div><strong>{{generated_time}}</strong></div>
                            </div>
                        </div>

                        <div style="display:flex; gap:14px; margin-block-start:14px">
                            <div style="flex:1">
                                <div class="card">
                                    <h3>Summary</h3>
                                    <div style="display:flex; gap:10px; align-items:center; justify-content:space-between">
                                        <div>
                                            <div style="color:var(--muted)">Modality</div>
                                            <div><strong>{{modality}}</strong></div>
                                        </div>
                                        <div>
                                            <div style="color:var(--muted)">QC</div>
                                            <div class="{{ 'qc-pass' if qc_passed else 'qc-fail' }}">{{ 'PASSED' if qc_passed else 'FAILED' }}</div>
                                        </div>
                                    </div>

                                    <div class="metrics">
                                        <div class="metric">
                                            <div style="color:var(--muted)">Mask ratio</div>
                                            <h2>{{mask_ratio if mask_ratio else 'N/A'}}</h2>
                                        </div>
                                        <div class="metric">
                                            <div style="color:var(--muted)">Uncertainty</div>
                                            <h2>{{uncertainty if uncertainty else 'N/A'}}</h2>
                                        </div>
                                        <div class="metric">
                                            <div style="color:var(--muted)">Warnings</div>
                                            <h2 style="font-size:0.9rem">{{warnings | join(', ') if warnings else 'None'}}</h2>
                                        </div>
                                    </div>

                                    <div style="margin-block-start:12px" class="downloads">
                                        <a href="/download/{{job_id}}/bundle.npz">Download .npz</a>
                                        <a href="/download/{{job_id}}/mask.npy">Mask (.npy)</a>
                                        <a href="/download/{{job_id}}/prob.npy">Probability (.npy)</a>
                                    </div>
                                </div>
                            </div>

                            <div style="flex:2">
                                <div class="card">
                                    <h3>Visualization</h3>
                                    {% if viz_b64 %}
                                        <img src="data:image/png;base64,{{viz_b64}}" class="viz"/>
                                    {% else %}
                                        <div style="color:var(--muted)">No visualization available</div>
                                    {% endif %}

                                    <h4 style="margin-block-start:12px">Details</h4>
                                    <table class="info">
                                        <tr><td>Job ID</td><td>{{job_id}}</td></tr>
                                        <tr><td>Modality</td><td>{{modality}}</td></tr>
                                        <tr><td>Mask ratio</td><td>{{mask_ratio if mask_ratio else 'N/A'}}</td></tr>
                                        <tr><td>Uncertainty</td><td>{{uncertainty if uncertainty else 'N/A'}}</td></tr>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </body>
                </html>
                '''

                # compute/display fields
                qc_passed = None
                warnings = []
                mask_ratio = None
                uncertainty = None

                # Prefer values stored in the job metadata if present
                if metadata.get('uncertainty') is not None:
                    uncertainty = metadata.get('uncertainty')
                elif isinstance(metadata.get('metadata'), dict) and metadata['metadata'].get('uncertainty') is not None:
                    uncertainty = metadata['metadata'].get('uncertainty')

                if isinstance(metadata.get('qc_results'), dict):
                    qc = metadata.get('qc_results', {})
                    qc_passed = qc.get('passed', None)
                    warnings = qc.get('warnings', [])
                    mask_ratio = qc.get('metrics', {}).get('mask_ratio', None)
                else:
                    # sometimes mask ratio is stored under metadata
                    if isinstance(metadata.get('metadata'), dict):
                        mask_ratio = metadata['metadata'].get('mask_ratio', None)

                # attempt to extract values from saved files if missing
                try:
                        bundle_file = os.path.join(job_dir, metadata.get('files', {}).get('bundle', 'bundle.npz'))
                        if os.path.exists(bundle_file):
                                arr = np.load(bundle_file)
                                if 'mask' in arr:
                                        m = arr['mask']
                                        mask_ratio = float(np.sum(m) / m.size)
                                if 'probability' in arr:
                                        p = arr['probability'].astype(np.float32)
                                        uncertainty = float(np.mean(p * (1.0 - p)))
                        else:
                                mask_file = os.path.join(job_dir, metadata.get('files', {}).get('mask', 'mask.npy'))
                                if os.path.exists(mask_file):
                                        m = np.load(mask_file)
                                        mask_ratio = float(np.sum(m) / m.size)
                                prob_file = os.path.join(job_dir, metadata.get('files', {}).get('probability_map', 'prob.npy'))
                                if os.path.exists(prob_file):
                                        p = np.load(prob_file).astype(np.float32)
                                        uncertainty = float(np.mean(p * (1.0 - p)))
                except Exception:
                        pass

                if qc_passed is None:
                        qc_passed = True if (mask_ratio is not None and mask_ratio > 0) else False

                # Render HTML and also save a preview copy into repository root for easy extraction
                rendered = render_template_string(
                    REPORT_HTML,
                    job_id=job_id,
                    modality=metadata.get('modality', 'UNKNOWN'),
                    qc_passed=qc_passed,
                    warnings=warnings,
                    mask_ratio=(None if mask_ratio is None else f"{mask_ratio:.4f}"),
                    uncertainty=(None if uncertainty is None else f"{uncertainty:.6f}"),
                    viz_b64=viz_b64,
                    generated_time=datetime.now().isoformat()
                )

                # Save a copy in the workspace so user can open/download it directly
                try:
                    preview_name = f"report_preview_{job_id}.html"
                    preview_path = os.path.join(os.getcwd(), preview_name)
                    with open(preview_path, 'w', encoding='utf-8') as pf:
                        pf.write(rendered)
                    print(f"✓ Saved report preview to {preview_path}")
                except Exception as e:
                    print(f"! Warning: could not save report preview: {e}")

                return rendered

    @app.route('/download/<job_id>/<filename>')
    def download(job_id, filename):
        job_dir = os.path.join(tempfile.gettempdir(), f"med_infer_{job_id}")
        file_path = os.path.join(job_dir, filename)
        if not os.path.exists(file_path):
            return 'File not found', 404
        return send_file(file_path, as_attachment=True)

    return app


def main():
    """Command-line interface for inference pipeline"""
    parser = argparse.ArgumentParser(
        description='Medical Image Segmentation - Inference Pipeline'
    )
    
    parser.add_argument('--model', type=str, default='best_model.pth',
                       help='Path to trained model weights (.pth file)')
    parser.add_argument('--input', type=str, required=False,
                       help='Path to input image or directory of images')
    parser.add_argument('--output', type=str, required=False,
                       help='Output directory for results')
    parser.add_argument('--serve', action='store_true',
                        help='Start a local web server for uploading images and getting reports')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for inference (default: cpu)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for segmentation (default: 0.5)')
    parser.add_argument('--uncertainty', action='store_true',
                       help='Enable uncertainty estimation (slower)')
    
    args = parser.parse_args()
    
    if args.serve:
        # Start web server
        app = create_app(model_path=args.model, device=args.device, threshold=args.threshold)
        print(f"Starting web server with model={args.model} on http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)
        return

    # Initialize pipeline
    pipeline = InferencePipeline(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.threshold
    )
    
    # Check if input is directory or single file
    input_path = Path(args.input)
    
    if input_path.is_dir():
        # Batch processing
        image_paths = list(input_path.glob('*.npy')) + \
                     list(input_path.glob('*.png')) + \
                     list(input_path.glob('*.jpg'))
        
        if len(image_paths) == 0:
            print(f"Error: No valid images found in {input_path}")
            return
        
        pipeline.batch_predict(
            image_paths=[str(p) for p in image_paths],
            output_dir=args.output
        )
    
    elif input_path.is_file():
        # Single file processing
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing single image: {input_path}")
        
        image = pipeline._load_image(str(input_path))
        results = pipeline.predict(image, uncertainty_estimation=args.uncertainty)
        
        # Save and visualize
        img_name = input_path.stem
        pipeline.visualize_results(
            image, results,
            save_path=str(output_dir / f"{img_name}_result.png")
        )
        
        # Save outputs
        np.save(output_dir / f"{img_name}_mask.npy", results['segmentation_mask'])
        np.save(output_dir / f"{img_name}_prob.npy", results['probability_map'])
        
        print(f"\n✓ Results saved to {output_dir}")
    
    else:
        print(f"Error: Invalid input path: {input_path}")


if __name__ == "__main__":
    main()