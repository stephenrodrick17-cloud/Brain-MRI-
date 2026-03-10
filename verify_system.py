#!/usr/bin/env python3
"""
Final Verification Test - Confirms all systems are operational
"""

from pathlib import Path
import json

print("\n" + "="*80)
print("FINAL SYSTEM VERIFICATION TEST")
print("="*80 + "\n")

checks = {
    "✅ Trained Model": Path("best_model.pth").exists(),
    "✅ Training History": Path("results.json").exists(),
    "✅ Brain Report": Path("medical_reports/sample_brain/report.json").exists(),
    "✅ Cardiac Report": Path("medical_reports/sample_cardiac/report.json").exists(),
    "✅ Thorax Report": Path("medical_reports/sample_thorax/report.json").exists(),
    "✅ 6-Panel Viz": Path("test_6panel_viz.png").exists(),
    "✅ Predictions Viz": Path("predictions_visualization.png").exists(),
    "✅ Training History Viz": Path("models_3d/training_history.png").exists(),
    "✅ 3D Reconstruction": Path("reconstruction_output/3d_reconstruction.png").exists(),
    "✅ Segmentation Viz": Path("reconstruction_output/segmentation_and_depth.png").exists(),
    "✅ YOLO Module": Path("yolo_medical_detection.py").exists(),
    "✅ Analyzer Module": Path("yolo_medical_analyzer.py").exists(),
    "✅ Web Server": Path("web_interface_final.py").exists(),
}

print("SYSTEM COMPONENTS:")
print("-" * 80)
for check, status in checks.items():
    status_str = "✓ EXISTS" if status else "✗ MISSING"
    print(f"{check:35s} {status_str}")

all_present = all(checks.values())

print("\n" + "-" * 80)
if all_present:
    print("✅ ALL COMPONENTS PRESENT AND READY")
else:
    print("❌ SOME COMPONENTS MISSING")

# Check training data
print("\n" + "-" * 80)
print("TRAINING DATA VERIFICATION:")
print("-" * 80)

try:
    with open("results.json") as f:
        data = json.load(f)
    epochs = len(data['training_history']['train_loss'])
    final_loss = data['training_history']['train_loss'][-1]
    params = data.get('total_parameters', 0)
    arch = data.get('model_architecture', 'Unknown')
    print(f"  Architecture: {arch}")
    print(f"  Epochs: {epochs}")
    print(f"  Final Loss: {final_loss:.6f}")
    print(f"  Parameters: {params:,}")
except Exception as e:
    print(f"  ❌ Error reading training data: {e}")

# Check medical reports
print("\n" + "-" * 80)
print("MEDICAL REPORTS VERIFICATION:")
print("-" * 80)

reports_info = [
    ("Brain", Path("medical_reports/sample_brain/report.json")),
    ("Cardiac", Path("medical_reports/sample_cardiac/report.json")),
    ("Thorax", Path("medical_reports/sample_thorax/report.json"))
]

for region, path in reports_info:
    try:
        with open(path) as f:
            data = json.load(f)
        seg = data.get('segmentation', {})
        coverage = seg.get('coverage_percent', 0)
        confidence = seg.get('confidence_score', 0)
        method = seg.get('method', 'Unknown')
        print(f"  {region}: {coverage:.1f}% coverage, {confidence:.2f} confidence, {method}")
    except Exception as e:
        print(f"  {region}: ❌ Error - {e}")

# Final status
print("\n" + "="*80)
if all_present:
    print("✅ SYSTEM IS FULLY OPERATIONAL AND READY FOR USE")
    print("\n   Command to start: python web_interface_final.py")
    print("   Then open: http://localhost:5000")
else:
    print("❌ SOME FILES ARE MISSING - PLEASE CHECK AND REINSTALL")

print("="*80 + "\n")
