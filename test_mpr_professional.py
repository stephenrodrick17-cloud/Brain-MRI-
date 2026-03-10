#!/usr/bin/env python3
"""Test Professional MRI System - Upload and Verify MPR Display"""

import requests
import numpy as np
from PIL import Image
import json
import time

# Create test image
print("📸 Creating test image...")
test_img = np.random.randint(50, 200, (200, 200), dtype=np.uint8)
test_img[60:140, 60:140] = 180  # Add bright region
test_img_pil = Image.fromarray(test_img)
test_img_pil.save('test_mri.jpg')
print("  ✓ Test image created: test_mri.jpg (200x200)")

# Upload
print("\n📤 Uploading to server...")
with open('test_mri.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'patient_name': 'Test Patient',
        'patient_id': 'TP-001',
        'patient_age': '45',
        'patient_gender': 'M',
        'scan_type': 'Brain',
        'scan_date': '2025-12-25',
        'field_strength': '3.0T'
    }
    
    try:
        response = requests.post('http://localhost:5000/process', files=files, data=data, timeout=30)
        print(f"  Status: HTTP {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ SUCCESS: {result['success']}")
            
            if result['success']:
                r = result['results']
                print(f"\n📊 RESULTS:")
                print(f"  Volume Dimensions: {r['volume_dimensions']}")
                print(f"  Volume Size: {r['volume_size']} voxels")
                print(f"  Mean Intensity: {r['mean_intensity']:.4f}")
                print(f"  Coverage: {r['volume_coverage']:.1f}%")
                
                mpr = result.get('mpr_image')
                if mpr:
                    print(f"\n🖼️  MPR Image:")
                    print(f"  ✓ Received: {len(mpr)} bytes")
                    print(f"  ✓ Format: {'data:image/png;base64' in mpr}")
                    print(f"\n✅ 6-PANEL MPR SHOULD NOW DISPLAY IN BROWSER")
                    print(f"\nOpen http://localhost:5000 and click 'Process MRI Scan' button")
                    print(f"Then switch to '6-Panel MPR' tab to see the visualization")
                else:
                    print("❌ No MPR image in response")
        else:
            print(f"  Error: {response.text}")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "="*70)
print("Check browser at http://localhost:5000")
print("="*70)
