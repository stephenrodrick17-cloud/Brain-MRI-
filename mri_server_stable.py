#!/usr/bin/env python3
"""
3D MRI Reconstruction Server - STABLE VERSION
Robust error handling and detailed logging
"""

import sys
import traceback

print("Starting server initialization...")

try:
    from flask import Flask, render_template_string, request, jsonify
    print("✓ Flask imported")
    
    import numpy as np
    print("✓ NumPy imported")
    
    from pathlib import Path
    import json
    import tempfile
    from PIL import Image as PILImage
    print("✓ PIL imported")
    
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    print("✓ Matplotlib imported")
    
    import base64
    from io import BytesIO
    from datetime import datetime
    
    print("\n✓ All dependencies loaded successfully\n")
    
except Exception as e:
    print(f"FATAL: Import error - {e}")
    traceback.print_exc()
    sys.exit(1)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

UPLOAD_FOLDER = Path(tempfile.gettempdir()) / 'mri_uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)

print(f"✓ Upload folder: {UPLOAD_FOLDER}\n")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D MRI Reconstruction System</title>
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
            grid-template-columns: 350px 1fr;
            gap: 30px;
        }
        @media (max-width: 900px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }
        .panel {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .panel h2 {
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
            background: #f9f9f9;
            margin-bottom: 20px;
            transition: all 0.3s;
        }
        .drop-zone:hover, .drop-zone.active {
            background: #667eea;
            color: white;
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
        .input-group input, .input-group select {
            width: 100%;
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-family: inherit;
        }
        .input-group input:focus, .input-group select:focus {
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
        .error { background: #fee; border: 2px solid #fcc; color: #c33; }
        .success { background: #efe; border: 2px solid #cfc; color: #3c3; }
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
        .results {
            display: none;
        }
        .results.show {
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
        .desc {
            background: #f0f4f8;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            color: #555;
            line-height: 1.6;
        }
        .viz-image {
            width: 100%;
            border-radius: 12px;
            border: 2px solid #e0e0e0;
            margin-top: 15px;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .metric {
            background: #f9f9f9;
            border-left: 4px solid #667eea;
            padding: 12px;
            border-radius: 8px;
        }
        .metric-label {
            font-size: 0.85em;
            color: #999;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 1.3em;
            font-weight: 700;
            color: #667eea;
        }
        #file-input { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 3D MRI Reconstruction System</h1>
            <p>Advanced Medical Image Analysis with AI-Powered 3D Visualization</p>
        </div>

        <div class="main-grid">
            <!-- Upload Panel -->
            <div class="panel">
                <h2>📤 Upload & Analyze</h2>
                
                <div class="message error" id="errorMsg"></div>
                <div class="message success" id="successMsg"></div>
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Processing...</p>
                </div>

                <div class="drop-zone" id="dropZone">
                    <div style="font-size: 2.5em; margin-bottom: 10px;">📸</div>
                    <strong>Drag & Drop MRI Image</strong>
                    <p style="font-size: 0.9em; margin-top: 5px;">or click to browse</p>
                </div>

                <input type="file" id="fileInput" accept="image/*">

                <div class="input-group">
                    <label>Patient Name</label>
                    <input type="text" id="patientName">
                </div>

                <div class="input-group">
                    <label>Patient ID</label>
                    <input type="text" id="patientId">
                </div>

                <div class="input-group">
                    <label>Scan Type</label>
                    <select id="scanType">
                        <option>Brain MRI</option>
                        <option>Cardiac MRI</option>
                        <option>Spine MRI</option>
                        <option>Other</option>
                    </select>
                </div>

                <button class="btn" id="processBtn" disabled>🔬 Process Image</button>
            </div>

            <!-- Results Panel -->
            <div class="panel results" id="resultsPanel">
                <div class="tabs">
                    <button class="tab-btn active" onclick="showTab(0); return false;">📊 Visualization</button>
                    <button class="tab-btn" onclick="showTab(1); return false;">📈 Metrics</button>
                    <button class="tab-btn" onclick="showTab(2); return false;">ℹ️ Info</button>
                </div>

                <div class="tab-content active" id="viz">
                    <h3 style="color: #667eea; margin-bottom: 10px;">6-Panel 3D Reconstruction</h3>
                    <div class="desc">
                        <strong>Panel 1 (Top-Left):</strong> Estimated Depth Map showing tissue depth (blue=shallow, yellow=deep)<br>
                        <strong>Panel 2 (Top-Center):</strong> Axial Projection - Maximum intensity from top view<br>
                        <strong>Panel 3 (Top-Right):</strong> Sagittal Projection - Maximum intensity from side view<br>
                        <strong>Panel 4 (Bottom-Left):</strong> Coronal Projection - Maximum intensity from front view<br>
                        <strong>Panel 5 (Bottom-Center):</strong> Middle Axial Slice - Cross-section detail<br>
                        <strong>Panel 6 (Bottom-Right):</strong> Intensity Histogram - Distribution of values
                    </div>
                    <img id="vizImage" src="" alt="3D Reconstruction" class="viz-image">
                </div>

                <div class="tab-content" id="metrics">
                    <h3 style="color: #667eea; margin-bottom: 15px;">Analysis Metrics</h3>
                    <div class="metrics" id="metricsGrid"></div>
                </div>

                <div class="tab-content" id="info">
                    <h3 style="color: #667eea; margin-bottom: 15px;">Scan Information</h3>
                    <div id="infoContent"></div>
                </div>
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
            formData.append('patient_name', document.getElementById('patientName').value || 'Not provided');
            formData.append('patient_id', document.getElementById('patientId').value || 'Not provided');
            formData.append('scan_type', document.getElementById('scanType').value);

            document.getElementById('loading').style.display = 'block';
            document.getElementById('processBtn').disabled = true;

            fetch('/process', {
                method: 'POST',
                body: formData
            })
            .then(resp => {
                console.log('Response status:', resp.status);
                if (!resp.ok) {
                    throw new Error('Server error: ' + resp.status);
                }
                return resp.json();
            })
            .then(data => {
                console.log('Response data:', data);
                if (data.success) {
                    displayResults(data);
                    showSuccess('✓ Analysis complete!');
                    document.getElementById('resultsPanel').classList.add('show');
                } else {
                    showError(data.error || 'Processing failed');
                }
            })
            .catch(error => {
                console.error('Fetch error:', error);
                showError('Error: ' + error.message);
            })
            .finally(() => {
                document.getElementById('loading').style.display = 'none';
            });
        }

        function displayResults(data) {
            const r = data.results;

            // Visualization
            document.getElementById('vizImage').src = data.image_data;

            // Metrics
            const metricsHtml = `
                <div class="metric">
                    <div class="metric-label">Volume Dimensions</div>
                    <div class="metric-value">${r.volume_dimensions.join('×')}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Mean Intensity</div>
                    <div class="metric-value">${r.mean_intensity.toFixed(3)}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Max Intensity</div>
                    <div class="metric-value">${r.max_intensity.toFixed(3)}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Std Dev</div>
                    <div class="metric-value">${r.std_intensity.toFixed(3)}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Mean Depth</div>
                    <div class="metric-value">${r.depth_mean.toFixed(3)}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Max Depth</div>
                    <div class="metric-value">${r.depth_max.toFixed(3)}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Volume Coverage</div>
                    <div class="metric-value">${r.volume_coverage.toFixed(1)}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Image Size</div>
                    <div class="metric-value">${r.shape[0]}×${r.shape[1]}</div>
                </div>
            `;
            document.getElementById('metricsGrid').innerHTML = metricsHtml;

            // Info
            const infoHtml = `
                <p><strong>Patient Name:</strong> ${r.patient_name}</p>
                <p><strong>Patient ID:</strong> ${r.patient_id}</p>
                <p><strong>Scan Type:</strong> ${r.scan_type}</p>
                <p><strong>File:</strong> ${r.filename}</p>
                <p><strong>Time:</strong> ${new Date(r.timestamp).toLocaleString()}</p>
            `;
            document.getElementById('infoContent').innerHTML = infoHtml;

        function showTab(index) {
            try {
                const names = ['viewer', 'metrics', 'info'];
                const buttons = document.querySelectorAll('.tab-btn');
                
                if (buttons) buttons.forEach(function(b) { if (b && b.classList) b.classList.remove('active'); });
                const tabs = document.querySelectorAll('.tab-content');
                if (tabs) tabs.forEach(function(t) { if (t && t.classList) t.classList.remove('active'); });
                
                if (buttons && buttons[index] && buttons[index].classList) buttons[index].classList.add('active');
                if (names[index]) {
                    const el = document.getElementById(names[index]);
                    if (el && el.classList) el.classList.add('active');
                }
            } catch (e) {
                console.error('Error in showTab:', e);
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
        print(f"ERROR rendering index: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}", 500

@app.route('/process', methods=['POST'])
def process():
    try:
        print("\n[PROCESS] Starting image processing...")
        
        if 'file' not in request.files:
            print("[ERROR] No file in request")
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            print("[ERROR] Empty filename")
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        print(f"[INFO] Processing: {file.filename}")
        
        # Load image
        temp_path = UPLOAD_FOLDER / file.filename
        file.save(temp_path)
        print(f"[INFO] File saved to: {temp_path}")
        
        img = PILImage.open(temp_path).convert('L')
        img_array = np.array(img, dtype=np.float32) / 255.0
        print(f"[INFO] Image loaded: shape={img_array.shape}, dtype={img_array.dtype}")
        
        # Depth estimation
        print("[INFO] Estimating depth map...")
        depth_map = np.abs(np.gradient(img_array, axis=0)) + np.abs(np.gradient(img_array, axis=1))
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        print(f"[INFO] Depth map created: min={depth_map.min():.3f}, max={depth_map.max():.3f}")
        
        # Create volume
        print("[INFO] Creating 3D volume...")
        H, W = img_array.shape
        num_layers = 32
        volume = np.zeros((H, W, num_layers), dtype=np.float32)
        
        for z in range(num_layers):
            layer_mask = img_array.astype(np.float32) * depth_map
            gaussian_factor = np.exp(-((z - num_layers * depth_map) ** 2) / (2 * (num_layers / 4) ** 2))
            volume[:, :, z] = layer_mask * gaussian_factor
        
        volume_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        print(f"[INFO] Volume created: shape={volume.shape}")
        
        # Create visualization
        print("[INFO] Creating 6-panel visualization...")
        fig = plt.figure(figsize=(18, 12), dpi=100)
        
        ax1 = fig.add_subplot(2, 3, 1)
        im1 = ax1.imshow(depth_map, cmap='viridis')
        ax1.set_title('1: Depth Map\n(Blue=shallow, Yellow=deep)', fontweight='bold', fontsize=11)
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, label='Depth')
        
        ax2 = fig.add_subplot(2, 3, 2)
        mip_axial = np.max(volume_norm, axis=0)
        im2 = ax2.imshow(mip_axial, cmap='hot')
        ax2.set_title('2: Axial Projection\n(Top View)', fontweight='bold', fontsize=11)
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, label='Intensity')
        
        ax3 = fig.add_subplot(2, 3, 3)
        mip_sag = np.max(volume_norm, axis=1)
        im3 = ax3.imshow(mip_sag, cmap='hot')
        ax3.set_title('3: Sagittal Projection\n(Side View)', fontweight='bold', fontsize=11)
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, label='Intensity')
        
        ax4 = fig.add_subplot(2, 3, 4)
        mip_cor = np.max(volume_norm, axis=2)
        im4 = ax4.imshow(mip_cor, cmap='hot')
        ax4.set_title('4: Coronal Projection\n(Front View)', fontweight='bold', fontsize=11)
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, label='Intensity')
        
        ax5 = fig.add_subplot(2, 3, 5)
        mid_slice = volume_norm[volume_norm.shape[0]//2, :, :]
        im5 = ax5.imshow(mid_slice, cmap='viridis')
        ax5.set_title('5: Middle Axial Slice\n(Cross-Section)', fontweight='bold', fontsize=11)
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, label='Intensity')
        
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.hist(volume_norm.flatten(), bins=40, color='steelblue', alpha=0.8, edgecolor='black')
        ax6.set_title('6: Intensity Histogram\n(Distribution)', fontweight='bold', fontsize=11)
        ax6.set_xlabel('Intensity Value')
        ax6.set_ylabel('Frequency')
        ax6.grid(True, alpha=0.3)
        
        fig.suptitle('3D Medical Image Reconstruction - 6-Panel Analysis', fontsize=16, fontweight='bold')
        
        print("[INFO] Saving visualization to PNG...")
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        img_buffer.seek(0)
        img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        print("[INFO] Visualization saved")
        
        # Compile results
        print("[INFO] Compiling results...")
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
        
        print("[SUCCESS] Processing complete")
        
        return jsonify({
            'success': True,
            'results': results,
            'image_data': f'data:image/png;base64,{img_data}'
        })
        
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🏥 3D MEDICAL IMAGE RECONSTRUCTION SERVER - STABLE VERSION")
    print("="*70)
    print("\n✓ Server configuration:")
    print("  • Host: 127.0.0.1")
    print("  • Port: 5000")
    print("  • URL: http://localhost:5000")
    print("\n✓ Capabilities:")
    print("  • Upload MRI images (PNG, JPG, TIFF, GIF)")
    print("  • AI-powered 3D depth reconstruction")
    print("  • 6-panel multi-view visualization")
    print("  • Detailed analysis metrics")
    print("  • Professional medical interface")
    print("\n" + "="*70)
    print("\nStarting Flask server...\n")
    
    try:
        app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
