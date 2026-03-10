#!/usr/bin/env python3
import numpy as np
from PIL import Image
import requests
import json

print("Creating test MRI image...")
arr = np.ones((256, 256), dtype=np.uint8) * 120
# Add a realistic pattern
for i in range(60, 196):
    for j in range(60, 196):
        dist = np.sqrt((i-128)**2 + (j-128)**2)
        if dist < 60:
            arr[i,j] = 200 - dist
        else:
            arr[i,j] = 100

img = Image.fromarray(arr)
img.save('c:/temp/mri_test.png')
print("✓ Test image created")

print("\nUploading to server...")
with open('c:/temp/mri_test.png', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/upload',
        files={'file': f},
        data={
            'patient_name': 'Test Patient',
            'patient_id': 'TP-001',
            'patient_age': '45',
            'scan_type': 'Brain'
        }
    )

print(f"Status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    if data.get('success'):
        print("\n✅ SUCCESS!")
        r = data['results']
        print(f"\nResults:")
        print(f"  Volume: {r['volume_dimensions']}")
        print(f"  Mean: {r['mean_intensity']:.2f}")
        print(f"  Coverage: {r['volume_coverage']:.1f}%")
        print(f"  Has MPR: {bool(data.get('mpr_image'))}")
    else:
        print(f"\n❌ Error: {data.get('error')}")
else:
    print(f"❌ HTTP Error: {response.status_code}")
    print(response.text[:500])
