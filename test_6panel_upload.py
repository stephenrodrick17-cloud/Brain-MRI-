#!/usr/bin/env python3
"""
Test the 6-panel visualization upload functionality
"""
import requests
import json
from pathlib import Path
import numpy as np
from PIL import Image
import time

# Create a test image
print("[*] Creating test brain MRI image...")
np.random.seed(42)
brain_data = np.random.randint(40, 180, (256, 256), dtype=np.uint8)

# Add brain-like structure
center = np.ogrid[:256, :256]
dist = np.sqrt((center[0]-128)**2 + (center[1]-128)**2)
mask = dist < 80
brain_data[mask] = np.clip(brain_data[mask] + 60, 0, 255).astype(np.uint8)

# Add some internal structure
inner_mask = dist < 50
brain_data[inner_mask] = np.clip(brain_data[inner_mask] + 40, 0, 255).astype(np.uint8)

test_img_path = Path('test_brain_mri.jpg')
Image.fromarray(brain_data).save(test_img_path)
print(f"[OK] Test image created: {test_img_path}")
print(f"     Shape: {brain_data.shape}, Intensity: {brain_data.min()}-{brain_data.max()}")

# Upload to server
print("\n[*] Uploading to server...")
try:
    with open(test_img_path, 'rb') as f:
        files = {'file': f}
        data = {
            'patient_name': 'Test Patient Brain',
            'patient_id': 'BRAIN001',
            'scan_type': 'Brain MRI T1',
            'findings': 'Test upload - 6 panel visualization'
        }
        
        response = requests.post('http://localhost:5000/process', files=files, data=data, timeout=30)
        result = response.json()
        
        if result.get('success'):
            print("[OK] Upload successful!")
            print("\nResults received:")
            print(json.dumps(result['results'], indent=2))
            
            # Check output files
            if 'viewer_url' in result:
                print(f"\n[OK] 3D Viewer URL: {result['viewer_url']}")
                print(f"[OK] View in browser: http://localhost:5000{result['viewer_url']}")
        else:
            print(f"[ERROR] Upload failed: {result.get('error')}")
            
except Exception as e:
    print(f"[ERROR] Connection failed: {e}")
    print("Make sure the server is running at http://localhost:5000")

print("\n[*] Test complete!")
