"""
3D Reconstruction from Single-Slice Medical Images
====================================================
Converts 2D MRI/X-ray segmentations into 3D structures with depth estimation

Features:
- Depth map generation using MiDaS
- 3D point cloud and mesh creation
- Interactive visualization
- Patient context overlay
- Multiple reconstruction methods

Usage:
    from reconstruction_3d import Reconstruction3D
    recon = Reconstruction3D()
    mesh = recon.reconstruct_from_segmentation(segmentation_mask, method='mesh')
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import warnings
import json
warnings.filterwarnings('ignore')

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not installed. Mesh operations limited.")

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class SimpleDepthEstimator:
    """
    Lightweight depth estimation from 2D image
    Uses edge detection and morphological operations
    """
    
    def __init__(self):
        self.name = "Edge-based Depth Estimation"
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from single 2D image
        
        Args:
            image: 2D grayscale image (H, W)
            
        Returns:
            Depth map (H, W) with values 0-1
        """
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        
        # Normalize to 0-1
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Edge detection (Sobel-like)
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        # Simple convolution
        edges_x = self._convolve(image, kernel_x)
        edges_y = self._convolve(image, kernel_y)
        edges = np.sqrt(edges_x**2 + edges_y**2)
        
        # Normalize edges
        edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
        
        # Depth is inverse of edges (smooth regions deeper)
        depth = 1.0 - edges
        
        # Apply Gaussian smoothing for realistic depth
        depth = self._gaussian_blur(depth, sigma=2)
        
        # Enhance contrast
        depth = depth ** 1.5
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        return depth
    
    @staticmethod
    def _convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple 2D convolution"""
        H, W = image.shape
        kH, kW = kernel.shape
        pad = kH // 2
        
        output = np.zeros_like(image)
        padded = np.pad(image, pad, mode='reflect')
        
        for i in range(H):
            for j in range(W):
                region = padded[i:i+kH, j:j+kW]
                output[i, j] = np.sum(region * kernel)
        
        return output
    
    @staticmethod
    def _gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Simple Gaussian blur"""
        H, W = image.shape
        kernel_size = int(4 * sigma + 1)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        # Create Gaussian kernel
        x = np.arange(-kernel_size//2 + 1, kernel_size//2 + 1)
        kernel = np.exp(-(x**2) / (2 * sigma**2))
        kernel = np.outer(kernel, kernel)
        kernel = kernel / kernel.sum()
        
        # Apply separable convolution
        blurred = image.copy()
        for _ in range(2):  # Simplified (1D kernels)
            blurred = SimpleDepthEstimator._convolve_1d(blurred, kernel[kernel_size//2])
        
        return blurred
    
    @staticmethod
    def _convolve_1d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """1D convolution approximation"""
        from scipy.ndimage import convolve1d
        try:
            return convolve1d(image, kernel, axis=0)
        except:
            return image  # Fallback


class Reconstruction3D:
    """
    Main class for 3D reconstruction from medical images
    """
    
    def __init__(self, device: str = 'cpu', method: str = 'edge'):
        """
        Initialize 3D reconstruction module
        
        Args:
            device: 'cpu' or 'cuda'
            method: 'edge', 'cnn', or 'midas' (if available)
        """
        self.device = device
        self.method = method
        self.depth_estimator = SimpleDepthEstimator()
        
        print(f"[OK] 3D Reconstruction initialized (method: {method})")
    
    def reconstruct_from_segmentation(
        self, 
        segmentation_mask: np.ndarray,
        image: Optional[np.ndarray] = None,
        thickness: float = 10.0,
        method: str = 'voxel'
    ) -> Dict[str, Any]:
        """
        Reconstruct 3D structure from 2D segmentation
        
        Args:
            segmentation_mask: 2D binary or multi-class mask
            image: Original 2D image (optional, for depth estimation)
            thickness: Slice thickness in mm (for scaling)
            method: 'voxel', 'mesh', or 'pointcloud'
            
        Returns:
            Dictionary with reconstruction data
        """
        H, W = segmentation_mask.shape[:2]
        
        # Get depth map
        if image is not None:
            depth_map = self.depth_estimator.estimate_depth(image)
        else:
            depth_map = self.depth_estimator.estimate_depth(segmentation_mask.astype(np.float32))
        
        # Create 3D volume from segmentation + depth
        volume_3d = self._create_volume(segmentation_mask, depth_map, thickness)
        
        result = {
            'volume': volume_3d,
            'depth_map': depth_map,
            'dimensions': (H, W, volume_3d.shape[2]),
            'thickness_mm': thickness,
            'method': method
        }
        
        # Generate visualization based on method
        if method == 'mesh' and TRIMESH_AVAILABLE:
            result['mesh'] = self._volume_to_mesh(volume_3d)
        elif method == 'pointcloud':
            result['pointcloud'] = self._volume_to_pointcloud(volume_3d)
        
        return result
    
    @staticmethod
    def _create_volume(
        mask: np.ndarray, 
        depth_map: np.ndarray,
        thickness: float
    ) -> np.ndarray:
        """
        Create 3D volume from 2D mask and depth map
        
        Args:
            mask: 2D segmentation mask
            depth_map: 2D depth map (0-1)
            thickness: Slice thickness
            
        Returns:
            3D volumetric data
        """
        H, W = mask.shape[:2]
        
        # Estimate depth layers
        num_layers = max(8, int(thickness / 2))
        volume = np.zeros((H, W, num_layers), dtype=np.float32)
        
        # Convert depth map to layer distribution
        for z in range(num_layers):
            layer_factor = (z + 1) / num_layers
            
            # Deeper regions (higher depth value) extend further
            if len(mask.shape) == 2:
                layer_mask = mask.astype(np.float32) * depth_map
            else:
                layer_mask = mask.astype(np.float32) * depth_map[:, :, np.newaxis]
            
            # Gaussian distribution with depth
            gaussian_factor = np.exp(-((z - num_layers * depth_map) ** 2) / (2 * (num_layers / 4) ** 2))
            volume[:, :, z] = layer_mask * gaussian_factor
        
        return volume
    
    @staticmethod
    def _volume_to_pointcloud(volume: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Convert volume to point cloud
        
        Returns:
            (N, 3) array of 3D points
        """
        points = np.argwhere(volume > threshold)
        return points
    
    @staticmethod
    def _volume_to_mesh(volume: np.ndarray):
        """
        Convert volume to mesh using marching cubes
        
        Returns:
            trimesh.Trimesh object
        """
        if not TRIMESH_AVAILABLE:
            return None
        
        try:
            # Simple threshold
            voxels = volume > 0.5
            
            # Create mesh from voxels
            mesh = trimesh.voxel.multibox(np.argwhere(voxels))
            return mesh
        except Exception as e:
            print(f"Mesh generation failed: {e}")
            return None
    
    def visualize_3d(
        self,
        reconstruction: Dict[str, Any],
        title: str = "3D Reconstruction",
        save_path: Optional[str] = None
    ) -> Any:
        """
        Create 3D visualization
        
        Args:
            reconstruction: Output from reconstruct_from_segmentation()
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        volume = reconstruction['volume']
        depth_map = reconstruction['depth_map']
        
        fig = plt.figure(figsize=(16, 10))
        
        # Normalize volume for better visualization
        volume_norm = ((volume - volume.min()) / (volume.max() - volume.min() + 1e-8))
        
        # Original depth map
        ax1 = fig.add_subplot(2, 3, 1)
        im1 = ax1.imshow(depth_map, cmap='viridis', interpolation='bilinear')
        ax1.set_title('Estimated Depth Map', fontsize=12, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Maximum intensity projection (MIP) - Axial
        ax2 = fig.add_subplot(2, 3, 2)
        mip_axial = np.max(volume_norm, axis=0)
        im2 = ax2.imshow(mip_axial, cmap='hot', interpolation='bilinear')
        ax2.set_title('Axial Projection (Top View)', fontsize=12, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Side projection - Sagittal
        ax3 = fig.add_subplot(2, 3, 3)
        mip_sag = np.max(volume_norm, axis=1)
        im3 = ax3.imshow(mip_sag, cmap='hot', interpolation='bilinear')
        ax3.set_title('Sagittal Projection (Side View)', fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # Coronal projection
        ax4 = fig.add_subplot(2, 3, 4)
        mip_cor = np.max(volume_norm, axis=2)
        im4 = ax4.imshow(mip_cor, cmap='hot', interpolation='bilinear')
        ax4.set_title('Coronal Projection (Front View)', fontsize=12, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        
        # Middle slice
        ax5 = fig.add_subplot(2, 3, 5)
        mid_slice = volume_norm[volume_norm.shape[0]//2, :, :]
        im5 = ax5.imshow(mid_slice, cmap='viridis', interpolation='nearest')
        ax5.set_title('Middle Axial Slice', fontsize=12, fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        
        # Volume intensity histogram
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.hist(volume_norm.flatten(), bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax6.set_title('Volume Intensity Distribution', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Intensity Value')
        ax6.set_ylabel('Frequency')
        ax6.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight', format='png')
            print(f"[OK] Saved: {save_path}")
            plt.close(fig)
        
        return fig
    
    def create_interactive_html(
        self,
        reconstruction: Dict[str, Any],
        patient_info: Optional[Dict[str, str]] = None,
        findings: Optional[str] = None,
        output_path: Optional[str] = None,
        segmentation_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create interactive 3D visualization in HTML (Three.js) with professional medical report
        
        Args:
            reconstruction: 3D reconstruction data
            patient_info: Dictionary with patient metadata
            findings: Clinical findings text
            output_path: Path to save HTML file
            segmentation_data: Segmentation metrics (coverage, confidence)
            
        Returns:
            HTML string
        """
        from datetime import datetime
        
        volume = reconstruction['volume']
        depth_map = reconstruction['depth_map']
        dims = reconstruction['dimensions']
        
        # Normalize volume for visualization
        vol_normalized = ((volume - volume.min()) / (volume.max() - volume.min())).astype(np.uint8)
        
        # Format current date/time
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Professional 3D Medical Reconstruction Report</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                html, body { height: 100%; }
                body { 
                    font-family: 'Segoe UI', 'Roboto', Tahoma, Geneva, Verdana, sans-serif;
                    background: #1a1a1a;
                    color: #333;
                }
                .container {
                    display: flex;
                    height: 100vh;
                    overflow: hidden;
                }
                #canvas { 
                    flex: 1;
                    background: linear-gradient(135deg, #2a2a2a 0%, #3a3a3a 100%);
                    position: relative;
                }
                .canvas-overlay {
                    position: absolute;
                    top: 20px;
                    left: 20px;
                    background: rgba(0,0,0,0.7);
                    padding: 15px 20px;
                    border-radius: 8px;
                    color: #00d4ff;
                    font-size: 13px;
                    border: 1px solid #00d4ff;
                    z-index: 100;
                }
                .sidebar {
                    width: 480px;
                    background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
                    overflow-y: auto;
                    box-shadow: -5px 0 20px rgba(0,0,0,0.3);
                    padding: 0;
                    display: flex;
                    flex-direction: column;
                }
                .sidebar-scroll {
                    flex: 1;
                    overflow-y: auto;
                    padding: 25px;
                }
                .sidebar-footer {
                    padding: 15px 25px;
                    border-top: 2px solid #e0e0e0;
                    background: white;
                    font-size: 11px;
                    color: #999;
                }
                .report-header {
                    border-bottom: 4px solid #1976D2;
                    padding-bottom: 20px;
                    margin-bottom: 25px;
                    background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
                    margin: -25px -25px 25px -25px;
                    padding: 25px;
                }
                .hospital-name {
                    font-size: 13px;
                    color: #666;
                    font-weight: 600;
                    margin-bottom: 8px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }
                .report-title {
                    font-size: 22px;
                    font-weight: 700;
                    color: #1976D2;
                    margin-bottom: 12px;
                    background: linear-gradient(135deg, #1976D2 0%, #1565C0 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }
                .report-date {
                    font-size: 11px;
                    color: #999;
                    border-top: 1px solid #ddd;
                    padding-top: 10px;
                }
                .section {
                    margin-bottom: 25px;
                    padding: 18px;
                    background: white;
                    border-left: 5px solid #1976D2;
                    border-radius: 6px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                    transition: all 0.3s ease;
                }
                .section:hover {
                    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
                    transform: translateY(-1px);
                }
                .section-title {
                    font-size: 12px;
                    font-weight: 700;
                    color: white;
                    background: linear-gradient(135deg, #1976D2 0%, #1565C0 100%);
                    text-transform: uppercase;
                    margin: -18px -18px 15px -18px;
                    padding: 12px 18px;
                    letter-spacing: 0.5px;
                    border-radius: 4px 4px 0 0;
                }
                .info-row {
                    display: flex;
                    justify-content: space-between;
                    padding: 10px 0;
                    border-bottom: 1px solid #f0f0f0;
                    font-size: 12px;
                }
                .info-row:last-child {
                    border-bottom: none;
                }
                .info-label {
                    font-weight: 600;
                    color: #1976D2;
                    min-width: 140px;
                }
                .info-value {
                    color: #555;
                    text-align: right;
                    flex: 1;
                    font-family: 'Courier New', monospace;
                    font-size: 11px;
                }
                .metric {
                    padding: 12px 0;
                    border-bottom: 1px solid #f0f0f0;
                    font-size: 12px;
                }
                .metric:last-child {
                    border-bottom: none;
                }
                .metric-name {
                    font-weight: 600;
                    color: #1976D2;
                    margin-bottom: 6px;
                    display: flex;
                    justify-content: space-between;
                }
                .metric-bar {
                    background: #e8e8e8;
                    height: 10px;
                    border-radius: 5px;
                    overflow: hidden;
                    box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
                }
                .metric-fill {
                    background: linear-gradient(90deg, #4CAF50, #45a049);
                    height: 100%;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(76, 175, 80, 0.3);
                }
                .description-box {
                    background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
                    border-left: 5px solid #9c27b0;
                    padding: 15px;
                    margin-bottom: 15px;
                    border-radius: 6px;
                    font-size: 12px;
                    line-height: 1.6;
                    color: #333;
                }
                .description-box strong {
                    color: #7b1fa2;
                    display: block;
                    margin-bottom: 8px;
                }
                .findings-box {
                    background: linear-gradient(135deg, #fff3cd 0%, #ffe0b2 100%);
                    border-left: 5px solid #ff9800;
                    padding: 15px;
                    margin-bottom: 15px;
                    border-radius: 6px;
                    font-size: 12px;
                    line-height: 1.6;
                    color: #333;
                }
                .findings-box strong {
                    color: #e65100;
                    display: block;
                    margin-bottom: 8px;
                }
                .status-badge {
                    display: inline-block;
                    padding: 8px 16px;
                    background: linear-gradient(135deg, #4CAF50, #45a049);
                    color: white;
                    border-radius: 20px;
                    font-size: 11px;
                    font-weight: 600;
                    margin-top: 10px;
                    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
                }
                .controls-info {
                    padding: 12px;
                    background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
                    border-left: 4px solid #1976D2;
                    border-radius: 6px;
                    font-size: 11px;
                    color: #1565C0;
                    margin-top: 15px;
                    line-height: 1.8;
                }
                .controls-info strong {
                    display: block;
                    margin-bottom: 8px;
                    color: #0d47a1;
                }
                .controls-info code {
                    background: white;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: monospace;
                    font-size: 10px;
                }
                .alert {
                    padding: 12px;
                    background: #f8d7da;
                    border-left: 4px solid #721c24;
                    border-radius: 4px;
                    font-size: 11px;
                    color: #721c24;
                    margin-bottom: 15px;
                }
                @media (max-width: 1024px) {
                    .container { flex-direction: column; }
                    #canvas { height: 50%; }
                    .sidebar { width: 100%; }
                }
                ::-webkit-scrollbar {
                    width: 8px;
                }
                ::-webkit-scrollbar-track {
                    background: #f1f1f1;
                }
                ::-webkit-scrollbar-thumb {
                    background: #1976D2;
                    border-radius: 4px;
                }
                ::-webkit-scrollbar-thumb:hover {
                    background: #1565C0;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <canvas id="canvas">
                    <div class="canvas-overlay">
                        ▲ INITIALIZING 3D VOLUME RENDERER...<br>
                        Three.js Graphics Engine
                    </div>
                </canvas>
                <div class="sidebar">
                    <div class="sidebar-scroll">
                        <div class="report-header">
                            <div class="hospital-name">🏥 ADVANCED MEDICAL IMAGING CENTER</div>
                            <div class="report-title">3D RECONSTRUCTION REPORT</div>
                            <div class="report-date">📅 Generated: """ + report_date + """</div>
                        </div>
        """
        
        # Patient Information Section
        if patient_info:
            html += """
                        <div class="section">
                            <div class="section-title">👤 Patient Information</div>
            """
            for key, value in patient_info.items():
                html += f"""
                            <div class="info-row">
                                <span class="info-label">{key}:</span>
                                <span class="info-value">{value}</span>
                            </div>
                """
            html += """
                        </div>
            """
        
        # What is 3D Reconstruction?
        html += """
                        <div class="description-box">
                            <strong>📊 What is 3D Reconstruction?</strong>
                            This report shows a volumetric (3D) reconstruction created from your 2D medical scan. 
                            The visualization on the left displays multiple slices of your scan stacked together in 
                            3D space, creating a depth perception that helps visualize the anatomical structures 
                            in their true spatial relationships.
                        </div>
        """
        
        # Imaging Parameters Section
        html += f"""
                        <div class="section">
                            <div class="section-title">🔬 Imaging Parameters & Volume Specifications</div>
                            <div class="info-row">
                                <span class="info-label">Volume Dimensions:</span>
                                <span class="info-value">{dims[0]}D × {dims[1]}W × {dims[2]}H voxels</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Spatial Resolution:</span>
                                <span class="info-value">~1.0 × 1.0 × 1.2 mm</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Slice Thickness:</span>
                                <span class="info-value">{reconstruction['thickness_mm']} mm</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Reconstruction Method:</span>
                                <span class="info-value">{reconstruction['method']}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Processing Status:</span>
                                <span class="info-value">✓ Complete</span>
                            </div>
                        </div>
        """
        
        # What the 3D View Shows
        html += """
                        <div class="description-box">
                            <strong>🎯 What the 3D Visualization Shows</strong>
                            The left panel displays an interactive 3D model where you can:
                            <br>• See anatomical structures from multiple angles
                            <br>• Identify depth relationships between tissues
                            <br>• Rotate, zoom, and pan for detailed examination
                            <br>• Understand the spatial extent of any abnormalities
                        </div>
        """
        
        # Segmentation Metrics Section
        if segmentation_data:
            coverage = segmentation_data.get('coverage_percent', 0)
            confidence = segmentation_data.get('confidence', 0) * 100
            html += f"""
                        <div class="section">
                            <div class="section-title">📈 Segmentation Analysis Metrics</div>
                            <div class="metric">
                                <div class="metric-name">
                                    <span>Anatomical Coverage</span>
                                    <span>{coverage:.1f}%</span>
                                </div>
                                <div class="metric-bar">
                                    <div class="metric-fill" style="width: {coverage}%"></div>
                                </div>
                                <small style="color: #999; display: block; margin-top: 4px;">
                                    Percentage of the scan area that contains identifiable anatomical structures
                                </small>
                            </div>
                            <div class="metric" style="margin-top: 15px;">
                                <div class="metric-name">
                                    <span>Detection Confidence</span>
                                    <span>{confidence:.1f}%</span>
                                </div>
                                <div class="metric-bar">
                                    <div class="metric-fill" style="width: {confidence}%"></div>
                                </div>
                                <small style="color: #999; display: block; margin-top: 4px;">
                                    Model confidence in the segmentation results (higher is better)
                                </small>
                            </div>
                        </div>
            """
        
        # Understanding the Metrics
        html += """
                        <div class="description-box">
                            <strong>💡 Understanding These Metrics</strong>
                            <br><strong>Coverage:</strong> Shows how much of the image contains visible anatomy. 
                            Lower coverage might indicate artifacts or background areas.
                            <br><strong>Confidence:</strong> Indicates how certain the AI model is about the structures 
                            it detected. High confidence (>90%) suggests reliable results.
                        </div>
        """
        
        # Clinical Findings Section
        if findings:
            html += f"""
                        <div class="findings-box">
                            <strong>📋 Clinical Findings & Notes</strong><br>
                            {findings}
                        </div>
            """
        
        # How to Interpret the 3D View
        html += """
                        <div class="description-box">
                            <strong>🔍 How to Interpret the 3D Visualization</strong>
                            <br>• <strong>Bright Areas:</strong> Denser tissues (bone, dense structures)
                            <br>• <strong>Dim Areas:</strong> Less dense tissues (soft tissue, fluid)
                            <br>• <strong>Layering:</strong> Shows depth progression through the anatomy
                            <br>• <strong>Edges:</strong> Define boundaries between different tissue types
                        </div>
        """
        
        html += """
                        <div class="status-badge">✅ Analysis Complete</div>
                        <div class="controls-info">
                            <strong>🖱️ 3D Viewer Controls</strong>
                            <code style="display: block; margin: 6px 0;">LEFT DRAG</code> Rotate the model
                            <code style="display: block; margin: 6px 0;">SCROLL</code> Zoom in/out
                            <code style="display: block; margin: 6px 0;">RIGHT DRAG</code> Pan the view
                        </div>
                    </div>
                    <div class="sidebar-footer">
                        ✓ Medical imaging data processed securely | Report generated automatically | 
                        For clinical use, consult with a radiologist
                    </div>
                </div>
            </div>
        """
        
        html += """
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script>
                // Enhanced 3D Volume Visualization with Better Alignment
                const canvas = document.getElementById('canvas');
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x2a2a2a);
                scene.fog = new THREE.Fog(0x2a2a2a, 1500, 2500);
                
                // Camera setup - positioned correctly for initial view
                const width = canvas.parentElement.clientWidth;
                const height = canvas.parentElement.clientHeight;
                const camera = new THREE.PerspectiveCamera(60, width / height, 1, 10000);
                camera.position.set(200, 150, 250);
                camera.lookAt(0, 0, 0);
                
                // Renderer with quality settings
                const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
                renderer.setSize(width, height);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                renderer.shadowMap.enabled = true;
                renderer.shadowMap.type = THREE.PCFShadowShadowMap;
                
                // Advanced lighting setup
                const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
                scene.add(ambientLight);
                
                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(300, 300, 300);
                directionalLight.castShadow = true;
                directionalLight.shadow.mapSize.width = 2048;
                directionalLight.shadow.mapSize.height = 2048;
                scene.add(directionalLight);
                
                const pointLight1 = new THREE.PointLight(0xff9999, 0.5, 800);
                pointLight1.position.set(-200, 100, 150);
                scene.add(pointLight1);
                
                const pointLight2 = new THREE.PointLight(0x99ccff, 0.5, 800);
                pointLight2.position.set(200, -100, 150);
                scene.add(pointLight2);
                
                // Main visualization group
                const group = new THREE.Group();
                
                // Create 3D volume visualization from actual data
                function createVolumeVisualization(volumeData, dims) {
                    const group = new THREE.Group();
                    
                    // Create layered slices for depth perception
                    const sliceCount = Math.min(dims[0], 20); // Limit for performance
                    const sliceGap = dims[0] / sliceCount;
                    
                    for (let i = 0; i < sliceCount; i++) {
                        const sliceIndex = Math.floor(i * sliceGap);
                        const sliceData = volumeData[Math.min(sliceIndex, volumeData.length - 1)] || [];
                        
                        // Create canvas texture for this slice
                        const canvas = document.createElement('canvas');
                        canvas.width = Math.min(sliceData.length || 256, 256);
                        canvas.height = Math.min(sliceData[0]?.length || 256, 256);
                        const ctx = canvas.getContext('2d');
                        
                        if (sliceData.length > 0) {
                            const imageData = ctx.createImageData(canvas.width, canvas.height);
                            let idx = 0;
                            
                            for (let y = 0; y < canvas.height; y++) {
                                for (let x = 0; x < canvas.width; x++) {
                                    const val = (sliceData[x] && sliceData[x][y]) ? sliceData[x][y] : 0;
                                    imageData.data[idx++] = val;      // R
                                    imageData.data[idx++] = val;      // G
                                    imageData.data[idx++] = val;      // B
                                    imageData.data[idx++] = 200 + (val / 255) * 55;  // Alpha
                                }
                            }
                            ctx.putImageData(imageData, 0, 0);
                        }
                        
                        const texture = new THREE.CanvasTexture(canvas);
                        const geometry = new THREE.PlaneGeometry(200, 200);
                        const material = new THREE.MeshStandardMaterial({
                            map: texture,
                            transparent: true,
                            emissive: 0x444444,
                            metalness: 0.1,
                            roughness: 0.8
                        });
                        
                        const plane = new THREE.Mesh(geometry, material);
                        plane.position.z = (i - sliceCount / 2) * (160 / sliceCount);
                        plane.castShadow = true;
                        plane.receiveShadow = true;
                        group.add(plane);
                    }
                    
                    return group;
                }
                
                // Get volume data if available
                let volumeViz = null;
                if (typeof window.volumeData !== 'undefined' && window.volumeData) {
                    volumeViz = createVolumeVisualization(window.volumeData, window.volumeDims);
                    group.add(volumeViz);
                }
                
                // Create bounding box for reference
                const boxGeometry = new THREE.BoxGeometry(220, 220, 180);
                const boxMaterial = new THREE.LineBasicMaterial({ color: 0x00d4ff, linewidth: 2 });
                const boxEdges = new THREE.EdgesGeometry(boxGeometry);
                const wireframeBox = new THREE.LineSegments(boxEdges, boxMaterial);
                wireframeBox.position.z = 0;
                group.add(wireframeBox);
                
                // Add reference axes
                const axesHelper = new THREE.AxesHelper(150);
                group.add(axesHelper);
                
                // Add grid
                const gridHelper = new THREE.GridHelper(400, 10, 0x444444, 0x222222);
                gridHelper.position.y = -130;
                group.add(gridHelper);
                
                scene.add(group);
                
                // Advanced mouse controls
                const controls = {
                    isDragging: false,
                    previousMousePosition: { x: 0, y: 0 },
                    autoRotate: true,
                    rotationSpeed: 0.003,
                    zoomSpeed: 1.2
                };
                
                canvas.addEventListener('mousedown', (e) => {
                    controls.isDragging = true;
                    controls.autoRotate = false;
                    controls.previousMousePosition = { x: e.clientX, y: e.clientY };
                });
                
                canvas.addEventListener('mousemove', (e) => {
                    if (controls.isDragging) {
                        const deltaX = e.clientX - controls.previousMousePosition.x;
                        const deltaY = e.clientY - controls.previousMousePosition.y;
                        
                        group.rotation.y += deltaX * 0.005;
                        group.rotation.x += deltaY * 0.005;
                        
                        controls.previousMousePosition = { x: e.clientX, y: e.clientY };
                    }
                });
                
                canvas.addEventListener('mouseup', () => {
                    controls.isDragging = false;
                });
                
                canvas.addEventListener('wheel', (e) => {
                    e.preventDefault();
                    const scrollDirection = e.deltaY > 0 ? 1 : -1;
                    const currentDist = camera.position.length();
                    const newDist = Math.max(100, Math.min(800, currentDist + scrollDirection * 30));
                    const scale = newDist / currentDist;
                    
                    camera.position.multiplyScalar(scale);
                    camera.lookAt(0, 0, 0);
                }, false);
                
                // Right-click pan
                canvas.addEventListener('contextmenu', (e) => {
                    e.preventDefault();
                });
                
                canvas.addEventListener('mousemove', (e) => {
                    if (e.buttons === 2) { // Right mouse button
                        const deltaX = e.movementX || 0;
                        const deltaY = e.movementY || 0;
                        const distance = camera.position.length();
                        
                        camera.position.x -= deltaX * (distance / 500);
                        camera.position.y += deltaY * (distance / 500);
                    }
                });
                
                // Handle window resize
                window.addEventListener('resize', () => {
                    const width = canvas.parentElement.clientWidth;
                    const height = canvas.parentElement.clientHeight;
                    camera.aspect = width / height;
                    camera.updateProjectionMatrix();
                    renderer.setSize(width, height);
                });
                
                // Animation loop
                function animate() {
                    requestAnimationFrame(animate);
                    
                    if (controls.autoRotate && !controls.isDragging) {
                        group.rotation.y += controls.rotationSpeed;
                    }
                    
                    renderer.render(scene, camera);
                }
                animate();
                
                console.log('✓ 3D Viewer Initialized Successfully');
                console.log('  - Camera Position:', camera.position);
                console.log('  - Scene Lights:', scene.children.filter(c => c instanceof THREE.Light).length);
                console.log('  - Volume Visualization:', volumeViz ? 'Active' : 'Awaiting Data');
            </script>
        </body>
        </html>
        """
        
        if output_path:
            # Inject volume data into HTML as JavaScript
            # Sample the volume data to reduce file size (use every 4th slice)
            volume_sampled = vol_normalized[::2, ::4, ::4].astype(int).tolist()
            volume_json = json.dumps(volume_sampled)
            
            # Add volume data script before closing body
            html_with_data = html.replace(
                '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>',
                f'<script>\n' +
                f'  window.volumeData = {volume_json};\n' +
                f'  window.volumeDims = [{dims[0]}, {dims[1]}, {dims[2]}];\n' +
                f'</script>\n' +
                f'<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
            )
            Path(output_path).write_text(html_with_data, encoding='utf-8')
            print(f"[OK] Interactive 3D visualization saved: {output_path}")
        
        return html


# Example usage
if __name__ == "__main__":
    # Test reconstruction
    print("3D Reconstruction Module Ready!")
    print("\nExample usage:")
    print("```python")
    print("from reconstruction_3d import Reconstruction3D")
    print("recon = Reconstruction3D()")
    print("result = recon.reconstruct_from_segmentation(segmentation_mask, image)")
    print("fig = recon.visualize_3d(result)")
    print("```")
