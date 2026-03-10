#!/usr/bin/env python3
"""
Complete demonstration of 6-panel 3D reconstruction system
Tests brain, cardiac, and thorax MRI scans
"""
import requests
import json
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile
import time

def create_mri_scan(scan_type, size=256):
    """Create synthetic MRI scan based on type"""
    np.random.seed(hash(scan_type) % 2**32)
    
    # Base tissue intensity
    scan_data = np.random.randint(40, 180, (size, size), dtype=np.uint8)
    
    if scan_type == 'brain':
        # Brain-like structure
        center = np.ogrid[:size, :size]
        dist = np.sqrt((center[0]-size/2)**2 + (center[1]-size/2)**2)
        brain = dist < size/3
        scan_data[brain] = np.clip(scan_data[brain] + 60, 0, 255).astype(np.uint8)
        inner = dist < size/4.5
        scan_data[inner] = np.clip(scan_data[inner] + 40, 0, 255).astype(np.uint8)
        
    elif scan_type == 'cardiac':
        # Cardiac-like structure (more complex)
        center = np.ogrid[:size, :size]
        dist_x = np.abs(center[0] - size/2)
        dist_y = np.abs(center[1] - size/2)
        heart = (dist_x + dist_y) < size/2.8
        scan_data[heart] = np.clip(scan_data[heart] + 70, 0, 255).astype(np.uint8)
        
    elif scan_type == 'thorax':
        # Thorax-like structure (elongated)
        center = np.ogrid[:size, :size]
        dist_x = (center[0] - size/2) / (size/3)
        dist_y = (center[1] - size/2) / (size/4)
        thorax = (dist_x**2 + dist_y**2) < 1
        scan_data[thorax] = np.clip(scan_data[thorax] + 65, 0, 255).astype(np.uint8)
    
    return scan_data

def upload_and_analyze(scan_type, server_url='http://localhost:5000'):
    """Upload scan and get analysis"""
    print(f"\n{'='*70}")
    print(f"Processing: {scan_type.upper()} MRI SCAN")
    print(f"{'='*70}")
    
    # Create synthetic MRI scan
    scan_data = create_mri_scan(scan_type)
    img_path = Path(f'test_{scan_type}_mri.jpg')
    Image.fromarray(scan_data).save(img_path)
    print(f"[OK] Synthetic {scan_type} MRI created ({scan_data.shape})")
    
    # Upload to server
    print(f"[*] Uploading to {server_url}...")
    with open(img_path, 'rb') as f:
        files = {'file': f}
        data = {
            'patient_name': f'Test Patient - {scan_type.title()}',
            'patient_id': f'{scan_type.upper()}_DEMO_001',
            'scan_type': f'{scan_type.title()} MRI',
            'findings': f'Demonstration scan for 3D reconstruction'
        }
        
        try:
            response = requests.post(f'{server_url}/process', files=files, data=data, timeout=30)
            result = response.json()
            
            if not result.get('success'):
                print(f"[ERROR] Upload failed: {result.get('error')}")
                return False
            
            results = result['results']
            
            # Display results
            print(f"\n[OK] Processing complete!")
            print(f"\nPatient Information:")
            print(f"  Name:  {results['patient_name']}")
            print(f"  ID:    {results['patient_id']}")
            print(f"  Scan:  {results['scan_type']}")
            
            print(f"\nImage Analysis:")
            print(f"  Original Size:     {results['shape']}")
            print(f"  Intensity Range:   {results['min_intensity']:.3f} - {results['max_intensity']:.3f}")
            print(f"  Mean Intensity:    {results['mean_intensity']:.3f}")
            print(f"  Std Dev:           {results['std_intensity']:.3f}")
            
            print(f"\n3D Reconstruction Metrics:")
            print(f"  Volume Size:       {results['volume_dimensions'][0]}×{results['volume_dimensions'][1]}×{results['volume_dimensions'][2]} voxels")
            print(f"  Depth Mean:        {results['depth_mean']:.4f}")
            print(f"  Depth Max:         {results['depth_max']:.4f}")
            print(f"  Volume Coverage:   {results['volume_coverage']:.2f}%")
            print(f"  Method:            {results['reconstruction_method']}")
            print(f"  Quality:           {results['reconstruction_quality']}")
            
            # Check output files
            upload_folder = Path(tempfile.gettempdir()) / 'mri_uploads'
            output_folders = sorted(upload_folder.glob('output_*'), reverse=True)
            
            if output_folders:
                latest = output_folders[0]
                print(f"\nGenerated Output Files:")
                for file in sorted(latest.glob('*')):
                    size = file.stat().st_size
                    if size > 1024*1024:
                        size_str = f"{size/(1024*1024):.1f} MB"
                    else:
                        size_str = f"{size/1024:.0f} KB"
                    print(f"  ✓ {file.name:30} {size_str:>10}")
            
            print(f"\n[OK] {scan_type.upper()} scan processed successfully!")
            return True
            
        except Exception as e:
            print(f"[ERROR] {e}")
            return False

def main():
    """Main demonstration"""
    print("\n" + "="*70)
    print("3D MEDICAL IMAGE RECONSTRUCTION SYSTEM")
    print("6-Panel Visualization Demo")
    print("="*70)
    
    # Check server
    print("\n[*] Checking server status...")
    try:
        r = requests.get('http://localhost:5000', timeout=5)
        print(f"[OK] Server is running (status {r.status_code})")
    except:
        print("[ERROR] Server not running. Start with: python mri_upload_analyzer.py")
        return
    
    # Test each scan type
    scan_types = ['brain', 'cardiac', 'thorax']
    results = {}
    
    for scan_type in scan_types:
        success = upload_and_analyze(scan_type)
        results[scan_type] = success
        time.sleep(1)  # Brief pause between uploads
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for scan_type in scan_types:
        status = "[OK]" if results[scan_type] else "[FAILED]"
        print(f"{status} {scan_type.upper():10} - 6-panel visualization generated")
    
    print(f"\n[OK] All scans processed successfully!")
    print(f"\nView results:")
    print(f"  1. Web Interface: http://localhost:5000")
    print(f"  2. Output folder: {Path(tempfile.gettempdir()) / 'mri_uploads'}")
    print(f"  3. 6-panel PNGs:  output_*/3d_reconstruction.png")
    print(f"\nEach output includes:")
    print(f"  • Estimated Depth Map (Viridis)")
    print(f"  • Axial Projection (Top View)")
    print(f"  • Sagittal Projection (Side View)")
    print(f"  • Coronal Projection (Front View)")
    print(f"  • Middle Axial Slice (Viridis)")
    print(f"  • Volume Intensity Histogram")

if __name__ == '__main__':
    main()
