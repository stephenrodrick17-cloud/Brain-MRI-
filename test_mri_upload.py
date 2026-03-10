#!/usr/bin/env python3
"""
Test upload script - creates test image and uploads to MRI system
"""
import numpy as np
from PIL import Image
import requests
import json
from datetime import datetime

print("="*70)
print("🧪 MRI SYSTEM TEST UPLOAD")
print("="*70)

# Create realistic test MRI image
print("\n1️⃣ Creating test MRI image...")
size = 256
test_img = Image.new('L', (size, size), color=100)
arr = np.array(test_img, dtype=np.float32)

# Create a realistic pattern (simulating MRI intensity variations)
for i in range(size):
    for j in range(size):
        # Center circle/blob (brain tissue)
        dist_to_center = np.sqrt((i - size/2)**2 + (j - size/2)**2)
        if dist_to_center < 80:
            arr[i, j] = 200 - dist_to_center * 0.5
        else:
            arr[i, j] = 100 + np.sin(i/30) * 20 + np.cos(j/30) * 20

arr = np.clip(arr, 0, 255).astype(np.uint8)
test_img = Image.fromarray(arr)
test_path = 'c:/temp/test_mri_upload.png'
test_img.save(test_path)
print(f"  ✓ Test image created: {test_path}")
print(f"  ✓ Size: {size}x{size}")

# Upload to server
print("\n2️⃣ Uploading to server...")
with open(test_path, 'rb') as f:
    files = {'file': f}
    data = {
        'patient_name': 'Test Patient',
        'patient_id': 'TP-2025-001',
        'patient_age': '45',
        'patient_gender': 'M',
        'scan_date': datetime.now().strftime('%Y-%m-%d'),
        'scan_type': 'Brain',
        'field_strength': '3.0 Tesla'
    }
    
    try:
        response = requests.post('http://localhost:5000/process', files=files, data=data, timeout=30)
        print(f"  ✓ Request sent")
        print(f"  ✓ Status code: {response.status_code}")
    except Exception as e:
        print(f"  ✗ Request failed: {e}")
        exit(1)

# Check response
print("\n3️⃣ Checking response...")
try:
    result = response.json()
    print(f"  ✓ JSON parsed successfully")
    
    if result.get('success'):
        print(f"\n✅ SUCCESS!")
        results = result.get('results', {})
        
        print(f"\n  📊 RESULTS:")
        print(f"    - Volume dimensions: {results.get('volume_dimensions')}")
        print(f"    - Volume size: {results.get('volume_size')} voxels")
        print(f"    - Mean intensity: {results.get('mean_intensity'):.2f}")
        print(f"    - Std intensity: {results.get('std_intensity'):.2f}")
        print(f"    - Coverage: {results.get('volume_coverage'):.1f}%")
        print(f"    - Patient: {results.get('patient_name')} (ID: {results.get('patient_id')})")
        
        mpr_img = result.get('mpr_image', '')
        print(f"\n  🖼️ MPR IMAGE:")
        if mpr_img:
            print(f"    ✓ Image generated: {len(mpr_img)} bytes")
            print(f"    ✓ Format: {mpr_img[:30]}...")
        else:
            print(f"    ✗ No image in response!")
        
        hist = results.get('intensity_histogram', [])
        print(f"\n  📈 HISTOGRAM:")
        print(f"    ✓ Bins: {len(hist)}")
        if hist:
            print(f"    ✓ Range: {min(hist):.3f} - {max(hist):.3f}")
        
    else:
        print(f"\n✗ ERROR: {result.get('error', 'Unknown error')}")
        
except json.JSONDecodeError as e:
    print(f"  ✗ JSON parse error: {e}")
    print(f"  Response text: {response.text[:500]}")

print("\n" + "="*70)
print("Test complete!")
print("="*70)
