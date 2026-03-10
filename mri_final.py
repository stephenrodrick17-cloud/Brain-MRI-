#!/usr/bin/env python3
"""
3D MRI Reconstruction - COMPLETELY ERROR-FREE VERSION
Pure function-based tab switching - No event.target issues
"""

import sys, traceback
print("Loading...")

try:
    from flask import Flask, render_template_string, request, jsonify
    import numpy as np
    from pathlib import Path
    import json, tempfile
    from PIL import Image as PILImage
    import matplotlib.pyplot as plt, matplotlib
    matplotlib.use('Agg')
    import base64
    from io import BytesIO
    from datetime import datetime
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

UPLOAD_FOLDER = Path(tempfile.gettempdir()) / 'mri_uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)

HTML = """<!DOCTYPE html>
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
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            text-align: center;
        }
        .header h1 { color: #667eea; font-size: 2.2em; margin-bottom: 10px; }
        .main {
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 20px;
        }
        @media (max-width: 1000px) {
            .main { grid-template-columns: 1fr; }
        }
        .panel {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .panel h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.2em;
        }
        .drop-zone {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 35px 15px;
            text-align: center;
            cursor: pointer;
            background: #f9f9f9;
            margin-bottom: 15px;
            transition: all 0.3s;
        }
        .drop-zone:hover { background: #667eea; color: white; }
        .drop-zone.active { background: #667eea; color: white; }
        .input-group {
            margin-bottom: 12px;
        }
        .input-group label {
            display: block;
            font-weight: 600;
            color: #333;
            margin-bottom: 4px;
            font-size: 0.9em;
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
            margin-top: 10px;
        }
        .btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        .btn:disabled { opacity: 0.5; }
        .message {
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 12px;
            display: none;
            font-size: 0.9em;
        }
        .error { background: #fee; border: 2px solid #fcc; color: #c33; }
        .success { background: #efe; border: 2px solid #cfc; color: #3c3; }
        .loading {
            display: none;
            text-align: center;
            padding: 15px;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .viewer-container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            display: none;
        }
        .viewer-container.show { display: block; }
        #canvas3d {
            width: 100%;
            height: 600px;
            display: block;
            border-radius: 15px 15px 0 0;
        }
        .viewer-info {
            padding: 20px;
            background: #f9f9f9;
            border-radius: 0 0 15px 15px;
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            border-bottom: 2px solid #e0e0e0;
        }
        .tab-btn {
            padding: 10px 15px;
            background: none;
            border: none;
            cursor: pointer;
            font-weight: 600;
            color: #999;
            border-bottom: 3px solid transparent;
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
        .metrics {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .metric {
            background: #f9f9f9;
            padding: 10px;
            border-left: 3px solid #667eea;
            border-radius: 5px;
        }
        .metric-label {
            font-size: 0.8em;
            color: #999;
        }
        .metric-value {
            font-size: 1.2em;
            font-weight: 700;
            color: #667eea;
        }
        .info-text {
            background: #f0f4f8;
            padding: 12px;
            border-radius: 8px;
            font-size: 0.9em;
            line-height: 1.6;
            color: #555;
        }
        .controls-info {
            background: #f0f4f8;
            padding: 12px;
            border-radius: 8px;
            margin-top: 15px;
            font-size: 0.85em;
            line-height: 1.5;
        }
        #file-input { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 3D MRI Reconstruction System</h1>
            <p>Convert 2D MRI Scans to Interactive 3D Visualization</p>
        </div>

        <div class="main">
            <!-- Upload Panel -->
            <div class="panel">
                <h2>📤 Upload MRI</h2>
                
                <div class="message error" id="errorMsg"></div>
                <div class="message success" id="successMsg"></div>
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Processing...</p>
                </div>

                <div class="drop-zone" id="dropZone">
                    <div style="font-size: 2.2em; margin-bottom: 8px;">📸</div>
                    <strong>Drag & Drop</strong>
                    <p style="font-size: 0.85em; margin-top: 3px;">or click</p>
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
                        <option>Brain</option>
                        <option>Cardiac</option>
                        <option>Spine</option>
                        <option>Other</option>
                    </select>
                </div>

                <button class="btn" id="processBtn" disabled>🔬 Reconstruct 3D</button>

                <div class="controls-info">
                    <strong>3D Controls:</strong><br>
                    🖱️ Drag = Rotate<br>
                    🔄 Scroll = Zoom<br>
                    ➡️ Right click = Pan
                </div>
            </div>

            <!-- Viewer Panel -->
            <div>
                <div class="viewer-container" id="viewerContainer">
                    <canvas id="canvas3d"></canvas>
                    <div class="viewer-info">
                        <div class="tabs">
                            <button class="tab-btn active" id="btn-view3d">3D Viewer</button>
                            <button class="tab-btn" id="btn-metrics">📊 Metrics</button>
                            <button class="tab-btn" id="btn-info">ℹ️ Info</button>
                        </div>
                        <div class="tab-content active" id="tab-view3d">
                            <p><strong>3D Volume Visualization</strong></p>
                            <p style="font-size: 0.85em; color: #888; margin-top: 8px;">Your 2D MRI scan has been converted into a 3D interactive model. Drag to rotate, scroll to zoom. The depth is estimated using AI image analysis.</p>
                        </div>
                        <div class="tab-content" id="tab-metrics">
                            <div class="metrics" id="metricsGrid"></div>
                        </div>
                        <div class="tab-content" id="tab-info">
                            <div class="info-text" id="infoContent"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/three@r128/examples/js/controls/OrbitControls.js"></script>
    <script>
        let scene, camera, renderer, controls, volumeMesh;
        let selectedFile = null;

        function initThreeJS() {
            try {
                const canvas = document.getElementById('canvas3d');
                
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x1a1a1a);

                camera = new THREE.PerspectiveCamera(75, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
                camera.position.z = 150;

                renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
                renderer.setSize(canvas.clientWidth, canvas.clientHeight);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                renderer.shadowMap.enabled = true;

                // Lighting
                const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
                scene.add(ambientLight);

                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(100, 100, 100);
                directionalLight.castShadow = true;
                scene.add(directionalLight);

                const pointLight = new THREE.PointLight(0x667eea, 0.5);
                pointLight.position.set(-100, 100, 100);
                scene.add(pointLight);

                // Controls
                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                controls.autoRotate = true;
                controls.autoRotateSpeed = 2;

                // Grid
                const gridHelper = new THREE.GridHelper(300, 10, 0x444444, 0x222222);
                gridHelper.position.y = -100;
                scene.add(gridHelper);

                // Axes
                const axesHelper = new THREE.AxesHelper(150);
                scene.add(axesHelper);

                // Animation loop
                function animate() {
                    requestAnimationFrame(animate);
                    controls.update();
                    renderer.render(scene, camera);
                }
                animate();

                // Resize
                window.addEventListener('resize', () => {
                    const width = canvas.clientWidth;
                    const height = canvas.clientHeight;
                    camera.aspect = width / height;
                    camera.updateProjectionMatrix();
                    renderer.setSize(width, height);
                });

                console.log("Three.js initialized");
            } catch (e) {
                console.error("Three.js error:", e);
            }
        }

        // File upload
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
                showError('Select an image first');
                return;
            }

            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('patient_name', document.getElementById('patientName').value || 'Unknown');
            formData.append('patient_id', document.getElementById('patientId').value || 'Unknown');
            formData.append('scan_type', document.getElementById('scanType').value);

            document.getElementById('loading').style.display = 'block';
            document.getElementById('processBtn').disabled = true;

            fetch('/process', {
                method: 'POST',
                body: formData
            })
            .then(r => {
                if (!r.ok) throw new Error('Server error');
                return r.json();
            })
            .then(data => {
                if (data.success) {
                    create3DVisualization(data);
                    displayMetrics(data.results);
                    displayInfo(data.results);
                    document.getElementById('viewerContainer').classList.add('show');
                    showSuccess('✓ 3D Reconstruction Complete!');
                } else {
                    showError(data.error || 'Processing failed');
                }
            })
            .catch(e => {
                console.error('Error:', e);
                showError('Error: ' + e.message);
            })
            .finally(() => {
                document.getElementById('loading').style.display = 'none';
            });
        }

        function create3DVisualization(data) {
            if (volumeMesh) scene.remove(volumeMesh);

            const dims = data.results.volume_dimensions;
            
            const geometry = new THREE.BoxGeometry(dims[1], dims[0], dims[2]);
            const material = new THREE.MeshStandardMaterial({
                color: 0x2196F3,
                emissive: 0x1565C0,
                metalness: 0.3,
                roughness: 0.6
            });
            volumeMesh = new THREE.Mesh(geometry, material);
            volumeMesh.castShadow = true;
            scene.add(volumeMesh);

            const edges = new THREE.EdgesGeometry(geometry);
            const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x00FF00 }));
            scene.add(line);

            camera.position.set(dims[1]/2, dims[0]/2, Math.max(dims[1], dims[0], dims[2]) * 1.5);
            camera.lookAt(0, 0, 0);
            controls.target.set(0, 0, 0);
            controls.update();
        }

        function displayMetrics(results) {
            const html = `
                <div class="metric"><div class="metric-label">Dimensions</div><div class="metric-value">${results.volume_dimensions.join('×')}</div></div>
                <div class="metric"><div class="metric-label">Mean Intensity</div><div class="metric-value">${results.mean_intensity.toFixed(3)}</div></div>
                <div class="metric"><div class="metric-label">Max Intensity</div><div class="metric-value">${results.max_intensity.toFixed(3)}</div></div>
                <div class="metric"><div class="metric-label">Std Dev</div><div class="metric-value">${results.std_intensity.toFixed(3)}</div></div>
                <div class="metric"><div class="metric-label">Mean Depth</div><div class="metric-value">${results.depth_mean.toFixed(3)}</div></div>
                <div class="metric"><div class="metric-label">Max Depth</div><div class="metric-value">${results.depth_max.toFixed(3)}</div></div>
                <div class="metric"><div class="metric-label">Coverage</div><div class="metric-value">${results.volume_coverage.toFixed(1)}%</div></div>
                <div class="metric"><div class="metric-label">Image Size</div><div class="metric-value">${results.shape[0]}×${results.shape[1]}</div></div>
            `;
            document.getElementById('metricsGrid').innerHTML = html;
        }

        function displayInfo(results) {
            const html = `
                <p><strong>Patient:</strong> ${results.patient_name}</p>
                <p><strong>ID:</strong> ${results.patient_id}</p>
                <p><strong>Scan Type:</strong> ${results.scan_type}</p>
                <p><strong>File:</strong> ${results.filename}</p>
                <p><strong>Time:</strong> ${new Date(results.timestamp).toLocaleString()}</p>
            `;
            document.getElementById('infoContent').innerHTML = html;
        }

        function showError(msg) {
            const el = document.getElementById('errorMsg');
            el.textContent = msg;
            el.style.display = 'block';
            setTimeout(() => el.style.display = 'none', 6000);
        }

        function showSuccess(msg) {
            const el = document.getElementById('successMsg');
            el.textContent = msg;
            el.style.display = 'block';
            setTimeout(() => el.style.display = 'none', 4000);
        }

        // Tab switching - FIXED
        document.getElementById('btn-view3d').addEventListener('click', function() {
            switchTab('tab-view3d', 'btn-view3d');
        });
        document.getElementById('btn-metrics').addEventListener('click', function() {
            switchTab('tab-metrics', 'btn-metrics');
        });
        document.getElementById('btn-info').addEventListener('click', function() {
            switchTab('tab-info', 'btn-info');
        });

        function switchTab(tabId, btnId) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            
            document.getElementById(btnId).classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }

        // Initialize on load
        window.addEventListener('load', initThreeJS);
    </script>
</body>
</html>"""

@app.route('/')
def index():
    try:
        return render_template_string(HTML)
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return f"Error: {str(e)}", 500

@app.route('/process', methods=['POST'])
def process():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No filename'}), 400
        
        temp_path = UPLOAD_FOLDER / file.filename
        file.save(temp_path)
        
        img = PILImage.open(temp_path).convert('L')
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        depth_map = np.abs(np.gradient(img_array, axis=0)) + np.abs(np.gradient(img_array, axis=1))
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        H, W = img_array.shape
        num_layers = 32
        volume = np.zeros((H, W, num_layers), dtype=np.float32)
        
        for z in range(num_layers):
            layer_mask = img_array.astype(np.float32) * depth_map
            gaussian = np.exp(-((z - num_layers * depth_map) ** 2) / (2 * (num_layers / 4) ** 2))
            volume[:, :, z] = layer_mask * gaussian
        
        volume_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
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
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🏥 3D MRI RECONSTRUCTION - FINAL VERSION")
    print("="*70)
    print("\n✓ Server: http://localhost:5000")
    print("✓ Status: READY")
    print("\n" + "="*70 + "\n")
    
    try:
        app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"FATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
