#!/usr/bin/env python3
"""Quick test upload without verbose output"""
print("Starting test...")
import requests
import numpy as np
from PIL import Image
import sys

try:
    print("Creating test image...")
    # Create test image
    test_img = np.random.randint(50, 200, (200, 200), dtype=np.uint8)
    test_img[60:140, 60:140] = 180
    test_img_pil = Image.fromarray(test_img)
    test_img_pil.save('test_mri.jpg')
    print("Test image created")
    
    # Upload
    print("Reading file...")
    with open('test_mri.jpg', 'rb') as f:
        files = {'file': f}
        data = {
            'patient_name': 'Test',
            'patient_id': 'T1',
            'patient_age': '45',
            'patient_gender': 'M',
            'scan_type': 'Brain',
            'scan_date': '2025-12-25',
            'field_strength': '3.0T'
        }
        
        print("Sending POST request...")
        response = requests.post('http://localhost:5000/upload', files=files, data=data, timeout=10)
        print(f'HTTP {response.status_code}')
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print('✓ Upload successful')
                print(f'✓ MPR image: {len(result["mpr_image"]) / 1024:.0f}KB')
            else:
                print(f'✗ Error: {result.get("error")}')
        else:
            print(f'✗ Error: {response.text[:200]}')
            
except Exception as e:
    print(f'✗ Exception: {e}')
    import traceback
    traceback.print_exc()
