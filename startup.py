#!/usr/bin/env python3
"""
FINAL INTEGRATED MEDICAL IMAGE ANALYSIS SYSTEM
Complete working system with all trained models integrated

QUICK START:
    python startup.py
    Then open: http://localhost:5000
"""

import subprocess
import sys
from pathlib import Path
import time
import webbrowser

def check_requirements():
    """Check if all required files exist"""
    print("\n" + "="*80)
    print("STARTUP VERIFICATION")
    print("="*80)
    
    required_files = {
        'models': [
            Path('best_model.pth'),
            Path('results.json'),
        ],
        'reports': [
            Path('medical_reports/sample_brain/report.json'),
            Path('medical_reports/sample_cardiac/report.json'),
            Path('medical_reports/sample_thorax/report.json'),
        ],
        'visualizations': [
            Path('test_6panel_viz.png'),
            Path('predictions_visualization.png'),
            Path('reconstruction_output/3d_reconstruction.png'),
        ],
        'code': [
            Path('web_interface_final.py'),
            Path('yolo_medical_detection.py'),
        ]
    }
    
    all_good = True
    for category, files in required_files.items():
        print(f"\n{category.upper()}:")
        for file in files:
            if file.exists():
                print(f"  ✓ {file}")
            else:
                print(f"  ❌ {file} - MISSING")
                all_good = False
    
    return all_good

def start_server():
    """Start the Flask server"""
    print("\n" + "="*80)
    print("STARTING SERVER")
    print("="*80)
    print("\nInitializing Medical Image Analysis System...")
    print("  • Loading trained models...")
    print("  • Loading medical reports...")
    print("  • Loading visualizations...")
    print("  • Initializing YOLO detector...\n")
    
    try:
        subprocess.run(
            [sys.executable, 'web_interface_final.py'],
            cwd=Path.cwd()
        )
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🏥 MEDICAL IMAGE ANALYSIS SYSTEM - FINAL BUILD")
    print("="*80)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Some required files are missing!")
        print("Please ensure all data files are in the correct location.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("✓ ALL REQUIREMENTS MET")
    print("="*80)
    
    print("""
SYSTEM READY TO START

Features:
  ✓ Trained Attention U-Net (50 epochs)
  ✓ 6-Panel MPR with Depth Analysis
  ✓ YOLO Tissue Detection
  ✓ Medical Report Generation
  ✓ REST API Interface

After starting, open: http://localhost:5000

Press Ctrl+C to stop the server
""")
    
    input("Press Enter to start the server...")
    start_server()
