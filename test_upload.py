#!/usr/bin/env python3
"""Test script for MRI upload and verification"""
import numpy as np
from PIL import Image
import requests
import json

# Create test image
test_img = Image.new('L', (256, 256), color=100)
# Add some pattern
arr = np.array(test_img)
for i in range(50, 200):
    for j in range(50, 200):
        arr[i, j] = 150 + np.sin(i/20) * 50 + np.cos(j/20) * 50
test_img = Image.fromarray(arr.astype(np.uint8))
test_img.save('/tmp/test_mri.png')

# Upload
print("Uploading test MRI image...")
with open('/tmp/test_mri.png', 'rb') as f:
    files = {'file': f}
    data = {
        'patient_name': 'Test Patient',
        'patient_id': 'TP-001',
        'patient_age': '45',
        'patient_gender': 'M',
        'scan_date': '2025-12-25',
        'scan_type': 'Brain',
        'field_strength': '3.0 Tesla'
    }
    
    response = requests.post('http://localhost:5000/process', files=files, data=data)
    
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

if response.status_code == 200:
    result = response.json()
    if result['success']:
        print("\n✓ SUCCESS!")
        print("Results returned:")
        print(f"  - Volume dimensions: {result['results']['volume_dimensions']}")
        print(f"  - Mean intensity: {result['results']['mean_intensity']:.2f}")
        print(f"  - Coverage: {result['results']['volume_coverage']:.1f}%")
        print(f"  - Has MPR image: {bool(result.get('mpr_image'))}")
    else:
        print(f"\n✗ FAILED: {result.get('error')}")
else:
    print(f"\n✗ Server error: {response.status_code}")
