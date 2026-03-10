"""
Web Interface for 3D Medical Image Reconstruction
==================================================
Flask application for uploading, segmenting, and visualizing 3D reconstructions

Features:
- Drag-and-drop image upload
- Real-time segmentation visualization
- Interactive 3D viewer
- Patient information management
- Report generation and download

Run:
    python web_interface_3d.py
    
Then open: http://localhost:5000
"""

from flask import Flask, render_template_string, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import torch
import tempfile
from pathlib import Path
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image as PILImage

from pipeline_segmentation_to_3d import SegmentationTo3D


# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)

# Configuration
UPLOAD_FOLDER = Path(tempfile.gettempdir()) / 'medical_images_3d'
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'dcm', 'tiff'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize pipeline
try:
    pipeline = SegmentationTo3D(
        model_path='best_model.pth',
        model_3d_path='models_3d/final_3d_model.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    pipeline_ready = True
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    pipeline_ready = False

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Medical Image Reconstruction</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
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
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 30px;
        }
        
        @media (max-width: 768px) {
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
            color: #333;
            font-size: 1.5em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        label {
            font-weight: 600;
            color: #555;
        }
        
        input, textarea {
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-family: inherit;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        
        input:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        textarea {
            resize: vertical;
            min-height: 100px;
        }
        
        .drop-zone {
            border: 3px dashed #667eea;
            border-radius: 5px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: #f8f9ff;
        }
        
        .drop-zone:hover {
            background: #f0f1ff;
            border-color: #764ba2;
        }
        
        .drop-zone.dragover {
            background: #e8e9ff;
            border-color: #764ba2;
        }
        
        .drop-zone p {
            color: #666;
            margin-bottom: 10px;
        }
        
        #file-input {
            display: none;
        }
        
        .button {
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            text-align: center;
            text-decoration: none;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-secondary {
            background: #f0f0f0;
            color: #333;
            border: 2px solid #ddd;
        }
        
        .btn-secondary:hover {
            background: #e0e0e0;
        }
        
        .preview {
            border-radius: 5px;
            overflow: hidden;
            background: #f5f5f5;
        }
        
        .preview img {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .results {
            margin-top: 20px;
        }
        
        .result-card {
            background: #f9f9f9;
            border-left: 4px solid #667eea;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        
        .result-card h3 {
            color: #333;
            margin-bottom: 8px;
        }
        
        .result-card p {
            color: #666;
            font-size: 0.95em;
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
        
        .error {
            background: #fee;
            border: 2px solid #fcc;
            color: #c33;
            padding: 15px;
            border-radius: 5px;
            display: none;
        }
        
        .success {
            background: #efe;
            border: 2px solid #cfc;
            color: #3c3;
            padding: 15px;
            border-radius: 5px;
            display: none;
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            border-bottom: 2px solid #ddd;
            margin-bottom: 20px;
        }
        
        .tab-btn {
            padding: 10px 20px;
            background: none;
            border: none;
            cursor: pointer;
            color: #666;
            font-weight: 600;
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
        
        .visualization {
            background: #f5f5f5;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }
        
        .visualization img {
            max-width: 100%;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 3D Medical Image Reconstruction</h1>
            <p>Transform 2D MRI/X-ray scans into interactive 3D visualizations</p>
        </div>
        
        <div class="content">
            <!-- Upload Section -->
            <div class="section">
                <h2>Upload Medical Image</h2>
                
                <div class="drop-zone" id="drop-zone">
                    <p>📁 Drag and drop your medical image here</p>
                    <p style="font-size: 0.9em; color: #999;">or click to select</p>
                    <input type="file" id="file-input" accept=".png,.jpg,.jpeg,.gif,.tiff">
                </div>
                
                <div id="file-preview" class="preview" style="display: none;">
                    <img id="preview-img" src="" alt="Preview">
                </div>
                
                <div class="error" id="error-message"></div>
                <div class="success" id="success-message"></div>
            </div>
            
            <!-- Patient Info & Findings -->
            <div class="section">
                <h2>Patient Information</h2>
                
                <div class="form-group">
                    <label for="patient-name">Patient Name</label>
                    <input type="text" id="patient-name" placeholder="e.g., John Doe">
                </div>
                
                <div class="form-group">
                    <label for="patient-id">Patient ID</label>
                    <input type="text" id="patient-id" placeholder="e.g., P123456">
                </div>
                
                <div class="form-group">
                    <label for="scan-type">Scan Type</label>
                    <input type="text" id="scan-type" placeholder="e.g., MRI Brain, Chest X-ray">
                </div>
                
                <div class="form-group">
                    <label for="findings">Clinical Findings</label>
                    <textarea id="findings" placeholder="Describe the findings observed in the scan..."></textarea>
                </div>
                
                <div id="loading" class="loading">
                    <div class="spinner"></div>
                    <p>Processing image and generating 3D reconstruction...</p>
                </div>
                
                <button class="button btn-primary" id="process-btn" onclick="processImage()" disabled>
                    ▶ Process & Reconstruct
                </button>
            </div>
        </div>
        
        <!-- Results Section -->
        <div id="results-section" style="display: none; padding: 30px; border-top: 2px solid #ddd;">
            <h2>Results & Visualizations</h2>
            
            <div class="tabs">
                <button class="tab-btn active" onclick="switchTabWeb(0); return false;">Segmentation</button>
                <button class="tab-btn" onclick="switchTabWeb(1); return false;">3D Reconstruction</button>
                <button class="tab-btn" onclick="switchTabWeb(2); return false;">3D Viewer</button>
                <button class="tab-btn" onclick="switchTabWeb(3); return false;">Report</button>
            </div>
            
            <!-- Segmentation Tab -->
            <div id="segmentation" class="tab-content active">
                <div class="result-card">
                    <h3>Segmentation Results</h3>
                    <p id="coverage-info"></p>
                    <p id="confidence-info"></p>
                </div>
                <div class="visualization">
                    <img id="segmentation-viz" src="" alt="Segmentation visualization">
                </div>
            </div>
            
            <!-- Reconstruction Tab -->
            <div id="reconstruction" class="tab-content">
                <div class="result-card">
                    <h3>3D Reconstruction Details</h3>
                    <p id="dimensions-info"></p>
                    <p id="method-info"></p>
                </div>
                <div class="visualization">
                    <img id="reconstruction-viz" src="" alt="3D reconstruction">
                </div>
            </div>
            
            <!-- 3D Viewer Tab -->
            <div id="viewer" class="tab-content">
                <div class="result-card">
                    <h3>Interactive 3D Viewer</h3>
                    <p>Click the button below to open the interactive 3D visualization in a new window.</p>
                </div>
                <button class="button btn-primary" onclick="openViewer()">🔍 Open Interactive 3D Viewer</button>
            </div>
            
            <!-- Report Tab -->
            <div id="report" class="tab-content">
                <div class="result-card">
                    <h3>Analysis Report</h3>
                    <pre id="report-content" style="background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto;"></pre>
                </div>
                <button class="button btn-primary" onclick="downloadReport()">⬇️ Download Report</button>
            </div>
            
            <div style="margin-top: 20px;">
                <button class="button btn-secondary" onclick="reset()">🔄 Process Another Image</button>
            </div>
        </div>
    </div>
    
    <script>
        let currentResults = null;
        let uploadedFile = null;
        
        // File upload handling
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        
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
            handleFileSelect(e.dataTransfer.files);
        });
        
        fileInput.addEventListener('change', (e) => {
            handleFileSelect(e.target.files);
        });
        
        function handleFileSelect(files) {
            if (files.length === 0) return;
            
            uploadedFile = files[0];
            
            // Validate file
            const allowedTypes = ['image/png', 'image/jpeg', 'image/gif', 'image/tiff'];
            if (!allowedTypes.includes(uploadedFile.type)) {
                showError('Invalid file type. Please upload PNG, JPG, GIF, or TIFF.');
                return;
            }
            
            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                document.getElementById('preview-img').src = e.target.result;
                document.getElementById('file-preview').style.display = 'block';
                document.getElementById('process-btn').disabled = false;
            };
            reader.readAsDataURL(uploadedFile);
        }
        
        function processImage() {
            if (!uploadedFile) {
                showError('Please select an image first.');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', uploadedFile);
            formData.append('patient_name', document.getElementById('patient-name').value);
            formData.append('patient_id', document.getElementById('patient-id').value);
            formData.append('scan_type', document.getElementById('scan-type').value);
            formData.append('findings', document.getElementById('findings').value);
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('process-btn').disabled = true;
            
            fetch('/process', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    currentResults = data;
                    displayResults(data);
                    showSuccess('Image processed successfully!');
                    document.getElementById('results-section').style.display = 'block';
                } else {
                    showError(data.error || 'Processing failed');
                }
            })
            .catch(error => {
                showError('Error: ' + error);
            })
            .finally(() => {
                document.getElementById('loading').style.display = 'none';
            });
        }
        
        function displayResults(data) {
            const results = data.results;
            
            // Segmentation info
            document.getElementById('coverage-info').textContent = 
                `Coverage: ${results.segmentation.coverage_percent.toFixed(2)}%`;
            document.getElementById('confidence-info').textContent = 
                `Confidence: ${results.segmentation.confidence_score.toFixed(4)}`;
            
            // Reconstruction info
            const dims = results.reconstruction.dimensions;
            document.getElementById('dimensions-info').textContent = 
                `Dimensions: ${dims[0]}×${dims[1]}×${dims[2]} voxels`;
            document.getElementById('method-info').textContent = 
                `Method: ${results.reconstruction.method}`;
            
            // Visualizations
            document.getElementById('segmentation-viz').src = data.images.segmentation;
            document.getElementById('reconstruction-viz').src = data.images.reconstruction;
            
            // Report
            document.getElementById('report-content').textContent = 
                JSON.stringify(results, null, 2);
        }
        
        function switchTabWeb(tabIndex) {
            try {
                const tabNames = ['segmentation', 'reconstruction', 'viewer', 'report'];
                const buttons = document.querySelectorAll('.tab-btn');
                
                if (buttons) buttons.forEach(function(btn) { if (btn && btn.classList) btn.classList.remove('active'); });
                const tabs = document.querySelectorAll('.tab-content');
                if (tabs) tabs.forEach(function(tab) { if (tab && tab.classList) tab.classList.remove('active'); });
                
                if (buttons && buttons[tabIndex] && buttons[tabIndex].classList) buttons[tabIndex].classList.add('active');
                if (tabNames[tabIndex]) {
                    const el = document.getElementById(tabNames[tabIndex]);
                    if (el && el.classList) el.classList.add('active');
                }
            } catch (e) {
                console.error('Error in switchTabWeb:', e);
            }
        }
        
        function openViewer() {
            if (!currentResults) {
                showError('No results available');
                return;
            }
            window.open(currentResults.viewer_url, '_blank', 'width=1200,height=800');
        }
        
        function downloadReport() {
            if (!currentResults) return;
            
            const dataStr = JSON.stringify(currentResults.results, null, 2);
            const dataBlob = new Blob([dataStr], { type: 'application/json' });
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'reconstruction_report.json';
            link.click();
        }
        
        function showError(msg) {
            const errorDiv = document.getElementById('error-message');
            errorDiv.textContent = msg;
            errorDiv.style.display = 'block';
            setTimeout(() => errorDiv.style.display = 'none', 5000);
        }
        
        function showSuccess(msg) {
            const successDiv = document.getElementById('success-message');
            successDiv.textContent = msg;
            successDiv.style.display = 'block';
            setTimeout(() => successDiv.style.display = 'none', 5000);
        }
        
        function reset() {
            location.reload();
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Main page"""
    if not pipeline_ready:
        return """
        <div style="padding: 50px; text-align: center; font-family: Arial;">
            <h2>⚠️ Model Not Loaded</h2>
            <p>Could not load the AI model. Please ensure 'best_model.pth' exists in the current directory.</p>
        </div>
        """
    return render_template_string(HTML_TEMPLATE)


@app.route('/process', methods=['POST'])
def process():
    """Process uploaded image"""
    if not pipeline_ready:
        return jsonify({'success': False, 'error': 'Pipeline not ready'})
    
    try:
        # Validate upload
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save uploaded file
        filename = secure_filename(file.filename or 'image.png')
        filepath = UPLOAD_FOLDER / filename
        file.save(filepath)
        
        # Get patient info
        patient_info = {}
        if request.form.get('patient_name'):
            patient_info['Name'] = request.form.get('patient_name')
        if request.form.get('patient_id'):
            patient_info['ID'] = request.form.get('patient_id')
        if request.form.get('scan_type'):
            patient_info['Scan Type'] = request.form.get('scan_type')
        
        findings = request.form.get('findings', '')
        
        # Process image with progress logging
        print(f"[Processing] Starting 3D reconstruction for: {filename}")
        results = pipeline.process_complete(
            str(filepath),
            patient_info=patient_info if patient_info else None,
            findings=findings if findings else None,
            output_dir=str(UPLOAD_FOLDER / 'output')
        )
        print(f"[Processing] 3D reconstruction complete!")
        
        # Convert images to base64
        seg_viz_path = Path(results['files']['segmentation_viz'])
        recon_viz_path = Path(results['files']['reconstruction_viz'])
        
        def img_to_base64(path):
            with open(path, 'rb') as f:
                return 'data:image/png;base64,' + base64.b64encode(f.read()).decode()
        
        response = {
            'success': True,
            'results': results['report'],
            'images': {
                'segmentation': img_to_base64(seg_viz_path),
                'reconstruction': img_to_base64(recon_viz_path)
            },
            'viewer_url': f'/viewer/{Path(results["files"]["3d_viewer"]).name}'
        }
        
        return app.response_class(
            response=json.dumps(response, cls=NumpyEncoder),
            status=200,
            mimetype='application/json'
        )
        
    except Exception as e:
        return app.response_class(
            response=json.dumps({'success': False, 'error': str(e)}, cls=NumpyEncoder),
            status=200,
            mimetype='application/json'
        )


@app.route('/viewer/<filename>')
def serve_viewer(filename):
    """Serve 3D viewer HTML"""
    return send_from_directory(UPLOAD_FOLDER / 'output', filename)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔬 3D Medical Image Reconstruction Web Interface")
    print("="*60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Model Ready: {pipeline_ready}")
    print("\nStarting server...")
    print("Open: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
