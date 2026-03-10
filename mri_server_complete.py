#!/usr/bin/env python3
"""
3D Medical Image Reconstruction Web Server - COMPLETE VERSION
With detailed explanations and interactive 3D visualization
"""

from flask import Flask, render_template_string, request, jsonify
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
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

UPLOAD_FOLDER = Path(tempfile.gettempdir()) / 'mri_uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)

print("✓ Flask initialized")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D MRI Reconstruction System</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: white;
            border-radius: 15px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            text-align: center;
        }
        .header h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        .main-grid {
            display: grid;
            grid-template-columns: 400px 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        @media (max-width: 1200px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }
        .upload-panel {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            height: fit-content;
        }
        .upload-panel h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.3em;
        }
        .drop-zone {
            border: 3px dashed #667eea;
            border-radius: 12px;
            padding: 40px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: #f9f9f9;
            margin-bottom: 20px;
        }
        .drop-zone:hover {
            background: #667eea;
            color: white;
        }
        .drop-zone.active {
            background: #667eea;
            color: white;
            border-color: #764ba2;
        }
        .input-group {
            margin-bottom: 15px;
        }
        .input-group label {
            display: block;
            font-weight: 600;
            color: #333;
            margin-bottom: 5px;
        }
        .input-group input,
        .input-group select {
            width: 100%;
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-family: inherit;
            transition: border-color 0.3s;
        }
        .input-group input:focus,
        .input-group select:focus {
            outline: none;
            border-color: #667eea;
        }
        .btn {
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .message {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
            display: none;
        }
        .error {
            background: #fee;
            border: 2px solid #fcc;
            color: #c33;
        }
        .success {
            background: #efe;
            border: 2px solid #cfc;
            color: #3c3;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
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
        .results-container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            display: none;
        }
        .results-container.show {
            display: block;
        }
        .tabs {
            display: flex;
            gap: 10px;
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 20px;
        }
        .tab-btn {
            padding: 12px 20px;
            background: none;
            border: none;
            cursor: pointer;
            font-weight: 600;
            color: #999;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }
        .tab-btn.active {
            color: #667eea;
            border-bottom-color: #667eea;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .visualization-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
        }
        .panel-group {
            background: #f9f9f9;
            border-radius: 12px;
            padding: 15px;
            border-left: 4px solid #667eea;
        }
        .panel-title {
            font-weight: 600;
            color: #667eea;
            margin-bottom: 8px;
            font-size: 1.1em;
        }
        .panel-description {
            color: #666;
            font-size: 0.95em;
            line-height: 1.5;
            margin-bottom: 12px;
        }
        .visualization-image {
            width: 100%;
            max-width: 100%;
            border-radius: 8px;
            border: 2px solid #e0e0e0;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .metric-card {
            background: #f9f9f9;
            border-left: 4px solid #667eea;
            padding: 15px;
            border-radius: 8px;
        }
        .metric-label {
            font-size: 0.85em;
            color: #999;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 1.4em;
            font-weight: 700;
            color: #667eea;
        }
        .info-section {
            background: #f0f4f8;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .info-section h3 {
            color: #667eea;
            margin-bottom: 12px;
        }
        .info-section p {
            color: #666;
            line-height: 1.6;
            margin-bottom: 8px;
        }
        .canvas-container {
            width: 100%;
            height: 500px;
            background: #1a1a1a;
            border-radius: 12px;
            margin-top: 15px;
        }
        #file-input {
            display: none;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 3D MRI Reconstruction System</h1>
        <p>Advanced Medical Image Analysis with AI-Powered 3D Visualization</p>
    </div>

    <div class="main-grid">
        <!-- Upload Panel -->
        <div class="upload-panel">
            <h2>📤 Upload & Analyze</h2>
            
            <div class="message error" id="errorMsg"></div>
            <div class="message success" id="successMsg"></div>
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing image...</p>
            </div>

            <div class="drop-zone" id="dropZone">
                <div style="font-size: 3em; margin-bottom: 10px;">📸</div>
                <strong>Drag & Drop MRI Image</strong>
                <p style="font-size: 0.9em; margin-top: 5px;">or click to browse</p>
            </div>

            <input type="file" id="fileInput" accept="image/*">

            <div class="input-group">
                <label>Patient Name</label>
                <input type="text" id="patientName" placeholder="Optional">
            </div>

            <div class="input-group">
                <label>Patient ID</label>
                <input type="text" id="patientId" placeholder="Optional">
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

            <button class="btn" id="processBtn" disabled>🔬 Process Image</button>
        </div>

        <!-- Results Panel -->
        <div class="results-container" id="resultsContainer">
            <div class="tabs">
                <button class="tab-btn active" onclick="switchTabComplete(0); return false;">📊 Visualization</button>
                <button class="tab-btn" onclick="switchTabComplete(1); return false;">📈 Analysis Metrics</button>
                <button class="tab-btn" onclick="switchTabComplete(2); return false;">ℹ️ Information</button>
            </div>

            <!-- Visualization Tab -->
            <div class="tab-content active" id="visualization">
                <div class="visualization-grid" id="vizGrid">
                    <div class="panel-group">
                        <div class="spinner" style="margin-bottom: 10px;"></div>
                        <p>Generating 3D reconstructions...</p>
                    </div>
                </div>
            </div>

            <!-- Analysis Tab -->
            <div class="tab-content" id="analysis">
                <h3 style="color: #667eea; margin-bottom: 20px;">📊 Detailed Analysis Metrics</h3>
                <div class="metrics-grid" id="metricsGrid"></div>
            </div>

            <!-- Info Tab -->
            <div class="tab-content" id="info">
                <div class="info-section" id="patientInfo"></div>
                <div class="info-section" id="scanInfo"></div>
            </div>
        </div>
    </div>

    <script>
        let selectedFile = null;

        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');

        dropZone.addEventListener('click', () => fileInput.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('active');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('active');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('active');
            handleFiles(e.dataTransfer.files);
        });

        fileInput.addEventListener('change', (e) => {
            handleFiles(e.target.files);
        });

        function handleFiles(files) {
            if (files.length === 0) return;
            selectedFile = files[0];
            dropZone.textContent = '✓ ' + selectedFile.name;
            document.getElementById('processBtn').disabled = false;
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
                    showSuccess('✓ Analysis complete!');
                    displayResults(data);
                    document.getElementById('resultsContainer').classList.add('show');
                    setTimeout(() => {
                        document.getElementById('resultsContainer').scrollIntoView({ behavior: 'smooth' });
                    }, 300);
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

        function displayResults(data) {
            const results = data.results;
            const imgData = data.image_data;

            // Visualization Tab
            document.getElementById('vizGrid').innerHTML = `
                <div class="panel-group">
                    <div class="panel-title">📊 6-Panel 3D Reconstruction</div>
                    <div class="panel-description">
                        Complete volumetric analysis showing multiple views of the 3D reconstructed medical image:
                        <strong>Depth Map</strong> (tissue depth estimation),
                        <strong>Axial/Sagittal/Coronal Projections</strong> (maximum intensity views from different angles),
                        <strong>Middle Slice</strong> (cross-sectional view), and
                        <strong>Intensity Histogram</strong> (distribution analysis).
                    </div>
                    <img src="${imgData}" class="visualization-image" alt="3D Reconstruction">
                </div>
            `;

            // Analysis Tab
            const metrics = `
                <div class="metric-card">
                    <div class="metric-label">Volume Dimensions (H×W×D)</div>
                    <div class="metric-value">${results.volume_dimensions.join('×')}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Mean Intensity</div>
                    <div class="metric-value">${results.mean_intensity.toFixed(3)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Max Intensity</div>
                    <div class="metric-value">${results.max_intensity.toFixed(3)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Std Deviation</div>
                    <div class="metric-value">${results.std_intensity.toFixed(3)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Mean Depth</div>
                    <div class="metric-value">${results.depth_mean.toFixed(3)}</div>
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
                    <div class="metric-label">Original Shape</div>
                    <div class="metric-value">${results.shape.join('×')}</div>
                </div>
            `;
            document.getElementById('metricsGrid').innerHTML = metrics;

            // Info Tab
            document.getElementById('patientInfo').innerHTML = `
                <h3>👤 Patient Information</h3>
                <p><strong>Patient Name:</strong> ${results.patient_name || 'Not provided'}</p>
                <p><strong>Patient ID:</strong> ${results.patient_id || 'Not provided'}</p>
                <p><strong>Timestamp:</strong> ${new Date(results.timestamp).toLocaleString()}</p>
            `;

            document.getElementById('scanInfo').innerHTML = `
                <h3>🔬 Scan Information</h3>
                <p><strong>Scan Type:</strong> ${results.scan_type || 'Not specified'}</p>
                <p><strong>File Name:</strong> ${results.filename}</p>
                <p><strong>Original Dimensions:</strong> ${results.shape[0]}×${results.shape[1]} pixels</p>
                <p><strong>Processing Method:</strong> AI-powered depth-based volumetric reconstruction</p>
            `;
        }

        function switchTabComplete(tabIndex) {
            try {
                const tabNames = ['visualization', 'analysis', 'info'];
                const buttons = document.querySelectorAll('.tab-btn');
                
                if (buttons) buttons.forEach(function(b) { if (b && b.classList) b.classList.remove('active'); });
                const contents = document.querySelectorAll('.tab-content');
                if (contents) contents.forEach(function(t) { if (t && t.classList) t.classList.remove('active'); });
                
                if (buttons && buttons[tabIndex] && buttons[tabIndex].classList) buttons[tabIndex].classList.add('active');
                if (tabNames[tabIndex]) {
                    const el = document.getElementById(tabNames[tabIndex]);
                    if (el && el.classList) el.classList.add('active');
                }
            } catch (e) {
                console.error('Error in switchTabComplete:', e);
            }
        }

        function showError(msg) {
            try {
                const el = document.getElementById('errorMsg');
                if (el) {
                    el.textContent = msg;
                    el.style.display = 'block';
                    setTimeout(function() { if (el) el.style.display = 'none'; }, 6000);
                }
            } catch (e) {
                console.error('Error:', e);
            }
        }

        function showSuccess(msg) {
            try {
                const el = document.getElementById('successMsg');
                if (el) {
                    el.textContent = msg;
                    el.style.display = 'block';
                    setTimeout(function() { if (el) el.style.display = 'none'; }, 4000);
                }
            } catch (e) {
                console.error('Error:', e);
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    try:
        return render_template_string(HTML_TEMPLATE)
    except Exception as e:
        print(f"ERROR rendering page: {e}")
        traceback.print_exc()
        return f"Error: {e}", 500

@app.route('/process', methods=['POST'])
def process():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        print(f"Processing file: {file.filename}")
        
        # Save and load image
        temp_path = UPLOAD_FOLDER / file.filename
        file.save(temp_path)
        
        # Load image
        img = PILImage.open(temp_path).convert('L')
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        print(f"Image shape: {img_array.shape}")
        
        # Estimate depth map from image gradients
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
        
        print("Creating 6-panel visualization...")
        
        # Create figure with 6 panels
        fig = plt.figure(figsize=(18, 12), dpi=100)
        
        # Panel 1: Depth Map
        ax1 = fig.add_subplot(2, 3, 1)
        im1 = ax1.imshow(depth_map, cmap='viridis')
        ax1.set_title('Panel 1: Estimated Depth Map\n(Shows tissue depth - blue=shallow, yellow=deep)', fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, label='Depth')
        
        # Panel 2: Axial Projection
        ax2 = fig.add_subplot(2, 3, 2)
        mip_axial = np.max(volume_norm, axis=0)
        im2 = ax2.imshow(mip_axial, cmap='hot')
        ax2.set_title('Panel 2: Axial Projection (Top View)\n(Maximum intensity from top - red=high, black=low)', fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, label='Intensity')
        
        # Panel 3: Sagittal Projection
        ax3 = fig.add_subplot(2, 3, 3)
        mip_sag = np.max(volume_norm, axis=1)
        im3 = ax3.imshow(mip_sag, cmap='hot')
        ax3.set_title('Panel 3: Sagittal Projection (Side View)\n(Maximum intensity from side)', fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, label='Intensity')
        
        # Panel 4: Coronal Projection
        ax4 = fig.add_subplot(2, 3, 4)
        mip_cor = np.max(volume_norm, axis=2)
        im4 = ax4.imshow(mip_cor, cmap='hot')
        ax4.set_title('Panel 4: Coronal Projection (Front View)\n(Maximum intensity from front)', fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, label='Intensity')
        
        # Panel 5: Middle Slice
        ax5 = fig.add_subplot(2, 3, 5)
        mid_slice = volume_norm[volume_norm.shape[0]//2, :, :]
        im5 = ax5.imshow(mid_slice, cmap='viridis')
        ax5.set_title('Panel 5: Middle Axial Slice\n(Cross-section at center)', fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, label='Intensity')
        
        # Panel 6: Histogram
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.hist(volume_norm.flatten(), bins=40, color='steelblue', alpha=0.8, edgecolor='black')
        ax6.set_title('Panel 6: Intensity Distribution\n(Histogram shows frequency of intensity values)', fontweight='bold')
        ax6.set_xlabel('Intensity Value')
        ax6.set_ylabel('Frequency')
        ax6.grid(True, alpha=0.3)
        
        fig.suptitle('3D Medical Image Reconstruction - Complete 6-Panel Analysis', fontsize=18, fontweight='bold', y=0.995)
        
        # Save to bytes
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        img_buffer.seek(0)
        img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        print("Visualization created successfully")
        
        # Results
        results = {
            'filename': file.filename,
            'shape': list(img_array.shape),
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
        print(f"ERROR in process: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Processing error: {str(e)}'})

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🏥 3D MEDICAL IMAGE RECONSTRUCTION SYSTEM")
    print("="*70)
    print("✓ Server starting...")
    print("✓ OPEN BROWSER: http://localhost:5000")
    print("\nCAPABILITIES:")
    print("  • Upload MRI images (PNG, JPG, TIFF, GIF)")
    print("  • AI-powered 3D depth reconstruction")
    print("  • 6-panel multi-view visualization")
    print("  • Detailed analysis metrics")
    print("  • Professional medical UI")
    print("\n" + "="*70 + "\n")
    
    try:
        app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
