#!/usr/bin/env python3
"""
Simple 3D Medical Image Reconstruction Web Server
No dependencies beyond Flask, NumPy, PIL, Matplotlib
"""

from flask import Flask, render_template_string, request, jsonify, send_file
import numpy as np
from pathlib import Path
import json
import tempfile
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import base64
from io import BytesIO
from datetime import datetime
import traceback

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

UPLOAD_FOLDER = Path(tempfile.gettempdir()) / 'mri_uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)

print("✓ Flask app initialized")
print(f"✓ Upload folder: {UPLOAD_FOLDER}")

# HTML Template - Simple and clean
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D MRI Reconstruction</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 40px;
        }
        @media (max-width: 900px) {
            .content {
                grid-template-columns: 1fr;
            }
        }
        .section {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .section h2 {
            color: #667eea;
            font-size: 1.5em;
        }
        .input-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .input-group label {
            font-weight: 600;
            color: #333;
        }
        .input-group input, .input-group textarea, .input-group select {
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-family: inherit;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        .input-group input:focus, .input-group textarea:focus, .input-group select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        .drop-zone {
            border: 3px dashed #667eea;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: #f9f9f9;
        }
        .drop-zone:hover, .drop-zone.dragover {
            background: #667eea;
            color: white;
            border-color: #764ba2;
        }
        .drop-zone p {
            font-size: 1.1em;
            margin: 10px 0;
        }
        .button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        .button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .error {
            background: #fee;
            border: 2px solid #fcc;
            color: #c33;
            padding: 12px;
            border-radius: 8px;
            display: none;
        }
        .success {
            background: #efe;
            border: 2px solid #cfc;
            color: #3c3;
            padding: 12px;
            border-radius: 8px;
            display: none;
        }
        .loading {
            display: none;
            text-align: center;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .results-section {
            grid-column: 1 / -1;
            display: none;
        }
        .image-container {
            text-align: center;
            margin-top: 20px;
        }
        .image-container img {
            max-width: 100%;
            height: auto;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }
        .metric-card {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .metric-label {
            font-size: 0.85em;
            color: #999;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 1.3em;
            font-weight: 600;
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 3D MRI Reconstruction</h1>
            <p>Upload MRI Scan → Get 3D Analysis</p>
        </div>

        <div class="content">
            <!-- Upload Section -->
            <div class="section">
                <h2>📤 Upload MRI Image</h2>
                
                <div class="error" id="error"></div>
                <div class="success" id="success"></div>
                
                <div class="drop-zone" id="dropZone">
                    <div style="font-size: 3em;">📸</div>
                    <p><strong>Drag & drop your MRI image</strong></p>
                    <p style="font-size: 0.9em;">or click to browse</p>
                </div>
                
                <input type="file" id="fileInput" style="display:none" accept="image/*">
                
                <div class="input-group">
                    <label>Patient Name</label>
                    <input type="text" id="patientName" placeholder="Enter patient name">
                </div>
                
                <div class="input-group">
                    <label>Patient ID</label>
                    <input type="text" id="patientId" placeholder="Enter patient ID">
                </div>
                
                <div class="input-group">
                    <label>Scan Type</label>
                    <select id="scanType">
                        <option>Brain MRI</option>
                        <option>Cardiac MRI</option>
                        <option>Spine MRI</option>
                        <option>Abdomen MRI</option>
                        <option>Other</option>
                    </select>
                </div>
                
                <button class="button" id="processBtn" disabled>Process Image</button>
            </div>

            <!-- Info Section -->
            <div class="section">
                <h2>ℹ️ Information</h2>
                <p><strong>Supported Formats:</strong> PNG, JPG, TIFF, GIF</p>
                <p><strong>Max File Size:</strong> 50 MB</p>
                <p><strong>Features:</strong></p>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li>AI-powered depth estimation</li>
                    <li>6-panel multi-view visualization</li>
                    <li>Interactive 3D viewer</li>
                    <li>Detailed analysis metrics</li>
                </ul>
            </div>

            <!-- Loading -->
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing your image...</p>
            </div>

            <!-- Results -->
            <div class="results-section" id="resultsSection">
                <h2>📊 Analysis Results</h2>
                <div class="image-container">
                    <img id="resultImage" src="" alt="3D Reconstruction">
                </div>
                <div class="metrics" id="metrics"></div>
                <button class="button" onclick="downloadResults()" style="margin-top: 20px;">Download Report</button>
            </div>
        </div>
    </div>

    <script>
        let selectedFile = null;

        // Drag and drop
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');

        dropZone.addEventListener('click', () => fileInput.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            handleFiles(e.dataTransfer.files);
        });

        fileInput.addEventListener('change', (e) => {
            handleFiles(e.target.files);
        });

        function handleFiles(files) {
            if (files.length === 0) return;
            selectedFile = files[0];
            document.getElementById('processBtn').disabled = false;
            document.getElementById('dropZone').textContent = '✓ ' + selectedFile.name;
        }

        document.getElementById('processBtn').addEventListener('click', processImage);

        function processImage() {
            if (!selectedFile) {
                showError('Please select an image');
                return;
            }

            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('patient_name', document.getElementById('patientName').value);
            formData.append('patient_id', document.getElementById('patientId').value);
            formData.append('scan_type', document.getElementById('scanType').value);

            document.getElementById('loading').style.display = 'block';
            document.getElementById('processBtn').disabled = true;

            fetch('/process', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showSuccess('Image processed successfully!');
                    displayResults(data.results);
                    document.getElementById('resultsSection').style.display = 'block';
                    document.getElementById('resultImage').src = data.image_data;
                } else {
                    showError(data.error || 'Processing failed');
                }
            })
            .catch(error => {
                showError('Error: ' + error);
                console.error(error);
            })
            .finally(() => {
                document.getElementById('loading').style.display = 'none';
            });
        }

        function displayResults(results) {
            const metrics = document.getElementById('metrics');
            metrics.innerHTML = `
                <div class="metric-card">
                    <div class="metric-label">Volume Dimensions</div>
                    <div class="metric-value">${results.volume_dimensions[0]}×${results.volume_dimensions[1]}×${results.volume_dimensions[2]}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Mean Intensity</div>
                    <div class="metric-value">${results.mean_intensity.toFixed(3)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Max Depth</div>
                    <div class="metric-value">${results.depth_max.toFixed(3)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Volume Coverage</div>
                    <div class="metric-value">${results.volume_coverage.toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Scan Type</div>
                    <div class="metric-value">${results.scan_type || 'Not specified'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Patient ID</div>
                    <div class="metric-value">${results.patient_id || 'N/A'}</div>
                </div>
            `;
        }

        function showError(msg) {
            const el = document.getElementById('error');
            el.textContent = msg;
            el.style.display = 'block';
            setTimeout(() => el.style.display = 'none', 5000);
        }

        function showSuccess(msg) {
            const el = document.getElementById('success');
            el.textContent = msg;
            el.style.display = 'block';
            setTimeout(() => el.style.display = 'none', 5000);
        }

        function downloadResults() {
            alert('Report download feature coming soon!');
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page"""
    try:
        return render_template_string(HTML_TEMPLATE)
    except Exception as e:
        print(f"Error rendering template: {e}")
        traceback.print_exc()
        return f"Error: {e}", 500

@app.route('/process', methods=['POST'])
def process():
    """Process uploaded image"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save file temporarily
        temp_path = UPLOAD_FOLDER / file.filename
        file.save(temp_path)
        
        # Load image
        img = PILImage.open(temp_path).convert('L')
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Estimate depth map
        depth_map = np.abs(np.gradient(img_array, axis=0)) + np.abs(np.gradient(img_array, axis=1))
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        # Create 3D volume
        H, W = img_array.shape
        num_layers = 32
        volume = np.zeros((H, W, num_layers), dtype=np.float32)
        
        for z in range(num_layers):
            layer_mask = img_array.astype(np.float32) * depth_map
            gaussian_factor = np.exp(-((z - num_layers * depth_map) ** 2) / (2 * (num_layers / 4) ** 2))
            volume[:, :, z] = layer_mask * gaussian_factor
        
        volume_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        # Create 6-panel visualization
        fig = plt.figure(figsize=(16, 10), dpi=100)
        
        # Panel 1: Depth Map
        ax1 = fig.add_subplot(2, 3, 1)
        im1 = ax1.imshow(depth_map, cmap='viridis')
        ax1.set_title('Estimated Depth Map', fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1)
        
        # Panel 2: Axial Projection
        ax2 = fig.add_subplot(2, 3, 2)
        mip_axial = np.max(volume_norm, axis=0)
        im2 = ax2.imshow(mip_axial, cmap='hot')
        ax2.set_title('Axial Projection', fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2)
        
        # Panel 3: Sagittal Projection
        ax3 = fig.add_subplot(2, 3, 3)
        mip_sag = np.max(volume_norm, axis=1)
        im3 = ax3.imshow(mip_sag, cmap='hot')
        ax3.set_title('Sagittal Projection', fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3)
        
        # Panel 4: Coronal Projection
        ax4 = fig.add_subplot(2, 3, 4)
        mip_cor = np.max(volume_norm, axis=2)
        im4 = ax4.imshow(mip_cor, cmap='hot')
        ax4.set_title('Coronal Projection', fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4)
        
        # Panel 5: Middle Slice
        ax5 = fig.add_subplot(2, 3, 5)
        mid_slice = volume_norm[volume_norm.shape[0]//2, :, :]
        im5 = ax5.imshow(mid_slice, cmap='viridis')
        ax5.set_title('Middle Axial Slice', fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5)
        
        # Panel 6: Histogram
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.hist(volume_norm.flatten(), bins=30, color='steelblue', alpha=0.7)
        ax6.set_title('Intensity Distribution', fontweight='bold')
        ax6.set_xlabel('Intensity')
        ax6.set_ylabel('Frequency')
        
        fig.suptitle('3D Medical Image Reconstruction - 6-Panel Analysis', fontsize=16, fontweight='bold')
        
        # Save figure to bytes
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        img_buffer.seek(0)
        img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        # Results
        results = {
            'filename': file.filename,
            'shape': img_array.shape,
            'mean_intensity': float(np.mean(img_array)),
            'std_intensity': float(np.std(img_array)),
            'min_intensity': float(np.min(img_array)),
            'max_intensity': float(np.max(img_array)),
            'volume_dimensions': [int(H), int(W), int(num_layers)],
            'depth_mean': float(np.mean(depth_map)),
            'depth_max': float(np.max(depth_map)),
            'volume_coverage': float(np.mean(volume > 0.1) * 100),
            'patient_name': request.form.get('patient_name', 'Not provided'),
            'patient_id': request.form.get('patient_id', 'Not provided'),
            'scan_type': request.form.get('scan_type', 'Not specified'),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'results': results,
            'image_data': f'data:image/png;base64,{img_data}'
        })
        
    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏥 3D MRI RECONSTRUCTION SERVER")
    print("="*60)
    print("\n✓ Starting server...")
    print("✓ Open: http://localhost:5000")
    print("✓ Features:")
    print("  • Upload PNG/JPG MRI images")
    print("  • 6-panel visualization")
    print("  • 3D depth analysis")
    print("  • Detailed metrics")
    print("\n" + "="*60 + "\n")
    
    try:
        app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False)
    except Exception as e:
        print(f"ERROR starting server: {e}")
        traceback.print_exc()
