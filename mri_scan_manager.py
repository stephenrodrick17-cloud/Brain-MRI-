#!/usr/bin/env python3
"""
Quick MRI Upload & Processing Tool
Easy interface for adding and processing MRI scans
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def setup_folders():
    """Create necessary folders for MRI processing"""
    folders = [
        'input_scans',
        'input_scans/brain',
        'input_scans/cardiac',
        'input_scans/spine',
        'results',
        'results/processed',
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"✓ Folder created/verified: {folder}")

def list_sample_images():
    """Show available sample images"""
    samples = [
        'sample_brain.jpg',
        'sample_cardiac.jpg',
        'sample_thorax.jpg',
    ]
    
    print("\n📸 Available Sample Images:")
    for sample in samples:
        path = Path(sample)
        if path.exists():
            size = path.stat().st_size / (1024*1024)  # Convert to MB
            print(f"  ✓ {sample} ({size:.2f} MB)")
        else:
            print(f"  ✗ {sample} (not found)")

def add_mri_scan(source_path, scan_type='brain', patient_id=''):
    """
    Add an MRI scan for processing
    
    Args:
        source_path: Path to MRI image file
        scan_type: Type of scan (brain, cardiac, spine)
        patient_id: Patient identifier
    """
    source = Path(source_path)
    
    if not source.exists():
        print(f"❌ Error: File not found: {source_path}")
        return False
    
    if scan_type not in ['brain', 'cardiac', 'spine']:
        print(f"❌ Error: Invalid scan type. Use: brain, cardiac, or spine")
        return False
    
    # Create destination
    dest_folder = Path(f'input_scans/{scan_type}')
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if patient_id:
        filename = f"{patient_id}_{timestamp}{source.suffix}"
    else:
        filename = f"{source.stem}_{timestamp}{source.suffix}"
    
    dest_path = dest_folder / filename
    
    # Copy file
    shutil.copy2(source, dest_path)
    
    size = dest_path.stat().st_size / (1024*1024)
    print(f"\n✅ MRI Scan Added Successfully!")
    print(f"   File: {filename}")
    print(f"   Type: {scan_type}")
    print(f"   Size: {size:.2f} MB")
    print(f"   Location: {dest_path}")
    
    return True

def show_processing_results():
    """Show previously processed results"""
    results_dir = Path('results/processed')
    
    if not results_dir.exists():
        print("No results found yet.")
        return
    
    print("\n📊 Processing Results:")
    for result_folder in sorted(results_dir.iterdir()):
        if result_folder.is_dir():
            viewer_file = result_folder / '3d_viewer.html'
            report_file = result_folder / 'report.json'
            
            print(f"\n  📁 {result_folder.name}")
            if viewer_file.exists():
                print(f"     ✓ Viewer: {viewer_file.name}")
            if report_file.exists():
                print(f"     ✓ Report: {report_file.name}")

def show_input_scans():
    """Show uploaded MRI scans waiting for processing"""
    print("\n📂 Uploaded MRI Scans:")
    
    input_dir = Path('input_scans')
    scan_types = ['brain', 'cardiac', 'spine']
    
    total_scans = 0
    for scan_type in scan_types:
        scan_folder = input_dir / scan_type
        if scan_folder.exists():
            scans = list(scan_folder.glob('*.jpg')) + list(scan_folder.glob('*.png')) + list(scan_folder.glob('*.jpeg'))
            if scans:
                print(f"\n  {scan_type.upper()} SCANS ({len(scans)}):")
                for scan in sorted(scans):
                    size = scan.stat().st_size / (1024*1024)
                    print(f"    • {scan.name} ({size:.2f} MB)")
                    total_scans += 1
    
    if total_scans == 0:
        print("\n  No scans uploaded yet.")
        print("  Use option 1 to add MRI scans.")

def print_menu():
    """Display main menu"""
    print("\n" + "="*60)
    print("🏥 MRI SCAN MANAGEMENT TOOL")
    print("="*60)
    print("\n1. Add MRI Scan")
    print("2. View Uploaded Scans")
    print("3. View Sample Images")
    print("4. View Processing Results")
    print("5. Setup Folders")
    print("6. Process Scans with Web Interface")
    print("7. Exit")
    print("\n" + "="*60)

def main():
    """Main menu loop"""
    print("\n🚀 Initializing MRI Scan Management System...")
    setup_folders()
    
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            print("\n📁 ADD NEW MRI SCAN")
            file_path = input("Enter path to MRI image: ").strip().strip('"')
            
            print("\nScan Type:")
            print("  1. Brain MRI")
            print("  2. Cardiac MRI")
            print("  3. Spine MRI")
            scan_choice = input("Select (1-3): ").strip()
            
            scan_types = {'1': 'brain', '2': 'cardiac', '3': 'spine'}
            scan_type = scan_types.get(scan_choice, 'brain')
            
            patient_id = input("Enter patient ID (optional): ").strip()
            
            add_mri_scan(file_path, scan_type, patient_id)
        
        elif choice == '2':
            show_input_scans()
        
        elif choice == '3':
            list_sample_images()
        
        elif choice == '4':
            show_processing_results()
        
        elif choice == '5':
            print("\n🔧 Setting up folders...")
            setup_folders()
        
        elif choice == '6':
            print("\n🌐 Opening Web Interface...")
            print("   Open your browser to: http://localhost:5000")
            print("   Use the web interface to:")
            print("   1. Upload MRI scans via drag-and-drop")
            print("   2. Enter patient information")
            print("   3. Process and view 3D reconstruction")
            print("   4. Download reports")
            input("\nPress Enter to continue...")
        
        elif choice == '7':
            print("\n👋 Thank you for using MRI Scan Management Tool!")
            break
        
        else:
            print("\n❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
