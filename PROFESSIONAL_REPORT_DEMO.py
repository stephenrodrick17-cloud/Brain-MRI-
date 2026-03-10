"""
Professional Medical 3D Reconstruction Report Demo
==================================================
This script demonstrates the complete professional medical report generation
with 3D interactive visualization.

Features:
- Professional medical report layout (like a doctor's report)
- Patient information
- Segmentation metrics with progress bars
- Clinical findings
- Interactive 3D viewer
- All details in the browser

Run:
    python PROFESSIONAL_REPORT_DEMO.py
"""

import subprocess
import sys
from pathlib import Path
import webbrowser
import time


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def create_sample_images():
    """Create sample medical images for demonstration"""
    import numpy as np
    from PIL import Image
    
    print("Creating sample medical images...")
    
    samples = [
        ("sample_brain.jpg", "Brain-like structure"),
        ("sample_cardiac.jpg", "Cardiac-like structure"),
        ("sample_thorax.jpg", "Thorax-like structure"),
    ]
    
    for filename, description in samples:
        size = 256
        img = np.zeros((size, size), dtype=np.uint8)
        
        # Create different patterns
        y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
        
        if "brain" in filename:
            # Brain-like
            mask = (x**2 + y**2) <= (30**2)
            img[mask] = 200
            mask = (x**2 + y**2) <= (80**2)
            img[mask] = 150
        elif "cardiac" in filename:
            # Heart-like shape
            mask1 = ((x-20)**2 + y**2) <= (40**2)
            mask2 = ((x+20)**2 + y**2) <= (40**2)
            mask3 = (x**2 + (y+30)**2) <= (50**2)
            img[mask1 | mask2 | mask3] = 180
        else:
            # Thorax-like
            mask = (x**2 + (y**2)/1.5) <= (80**2)
            img[mask] = 160
        
        # Add noise
        noise = np.random.randint(0, 15, (size, size))
        img = img + noise
        img = np.clip(img, 0, 255)
        
        # Save
        pil_img = Image.fromarray(img.astype(np.uint8))
        pil_img.save(filename)
        print(f"  ✓ {filename} - {description}")


def process_image(image_path, patient_name, patient_id, findings):
    """Process a single image and generate report"""
    print(f"\nProcessing: {image_path}")
    print("-" * 60)
    
    cmd = [
        sys.executable,
        'pipeline_segmentation_to_3d.py',
        '--image', image_path,
        '--model', 'best_model.pth',
        '--model-3d', 'models_3d/final_3d_model.pth',
        '--patient-name', patient_name,
        '--patient-id', patient_id,
        '--findings', findings,
        '--output', f'./medical_reports/{Path(image_path).stem}'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Extract the output directory from the result
        output_dir = f'./medical_reports/{Path(image_path).stem}'
        html_file = f'{output_dir}/3d_viewer.html'
        print(f"\n✓ Report generated successfully!")
        print(f"  View report: {html_file}")
        return html_file
    else:
        print(f"✗ Error processing {image_path}")
        print(result.stderr)
        return None


def main():
    print_header("Professional Medical 3D Reconstruction Report Generator")
    
    # Create medical reports directory
    Path('medical_reports').mkdir(exist_ok=True)
    
    # Step 1: Create sample images
    print("\n" + "="*70)
    print("  STEP 1: Creating Sample Medical Images")
    print("="*70)
    create_sample_images()
    
    # Step 2: Process images with detailed patient information
    print("\n" + "="*70)
    print("  STEP 2: Generating Professional Medical Reports")
    print("="*70)
    
    cases = [
        {
            'image': 'sample_brain.jpg',
            'name': 'John Doe',
            'id': 'MRI-BR-2025-001',
            'findings': 'Brain MRI shows normal anatomical structures. Ventricular system is symmetric. No focal lesions or abnormal signal intensity. No evidence of acute intracranial pathology. Sagittal midline structures are normal. Overall impression: NORMAL STUDY.'
        },
        {
            'image': 'sample_cardiac.jpg',
            'name': 'Jane Smith',
            'id': 'CT-CARD-2025-002',
            'findings': 'Cardiac CT imaging demonstrates normal heart size and configuration. Left and right ventricles are preserved. No pericardial effusion. Coronary arteries are patent without significant stenosis. Myocardial thickness is normal. Impression: NO ACUTE CARDIAC PATHOLOGY.'
        },
        {
            'image': 'sample_thorax.jpg',
            'name': 'Michael Johnson',
            'id': 'XR-THORAX-2025-003',
            'findings': 'Chest radiograph shows clear lung fields bilaterally. Heart size is normal. Mediastinal contours are unremarkable. No pleural effusion. Costophrenic angles are sharp. Bony thorax is intact. Impression: NORMAL CHEST RADIOGRAPH.'
        }
    ]
    
    reports = []
    for case in cases:
        html_file = process_image(
            case['image'],
            case['name'],
            case['id'],
            case['findings']
        )
        if html_file:
            reports.append(html_file)
            time.sleep(1)  # Small delay between processing
    
    # Step 3: Summary
    print_header("Report Generation Complete!")
    
    print("Generated Medical Reports:")
    for i, report in enumerate(reports, 1):
        print(f"\n  {i}. {report}")
    
    print("\n" + "="*70)
    print("  WHAT YOU GET:")
    print("="*70)
    print("""
  ✓ Professional Medical Report Layout
    - Hospital header with timestamp
    - Patient demographics
    - Segmentation metrics with visual bars
    - Clinical findings section
    - Volume reconstruction parameters
    - Status badge showing analysis complete
    
  ✓ Interactive 3D Viewer
    - Drag to rotate the 3D volume
    - Scroll to zoom in/out
    - Right-click to pan
    - Real-time rendering in WebGL
    
  ✓ All Details Visible in One View
    - No need to open separate JSON files
    - Professional layout suitable for printing
    - All metrics and findings on one screen
    """)
    
    print("\n" + "="*70)
    print("  HOW TO VIEW REPORTS:")
    print("="*70)
    print("""
  1. Each report is in: medical_reports/<case_name>/3d_viewer.html
  
  2. Open any report in your browser:
     - Double-click the .html file, OR
     - Right-click → Open with → Your browser
     
  3. The report shows:
     - 3D reconstruction on the left
     - Professional medical report on the right
     - Fully interactive and zoomable
    """)
    
    # Offer to open first report
    if reports:
        response = input("\nWould you like to open the first report now? (y/n): ").lower()
        if response == 'y':
            first_report = Path(reports[0]).resolve()
            print(f"\nOpening: {first_report}")
            webbrowser.open(f'file:///{first_report}')
            print("✓ Report opened in your default browser")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Demo cancelled by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
