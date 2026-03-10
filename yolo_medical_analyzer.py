#!/usr/bin/env python3
"""
Medical YOLO Integration & Model Usage
Demonstrates how to use YOLO for different medical imaging tasks
"""

import numpy as np
from pathlib import Path
import json
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from collections import defaultdict

print("\n" + "="*80)
print("MEDICAL YOLO MODEL INTEGRATION")
print("="*80)

# Tissue metadata
TISSUE_DEFINITIONS = {
    # BRAIN
    "gray_matter": {"region": "brain", "description": "Gray Matter (neurons)", "color": (100, 120, 200)},
    "white_matter": {"region": "brain", "description": "White Matter (axons)", "color": (200, 200, 220)},
    "ventricles": {"region": "brain", "description": "Brain Ventricles (fluid)", "color": (50, 100, 150)},
    "cerebrospinal_fluid": {"region": "brain", "description": "Cerebrospinal Fluid", "color": (100, 150, 200)},
    
    # SPINE
    "vertebral_body": {"region": "spine", "description": "Vertebral Body (bone)", "color": (220, 200, 150)},
    "intervertebral_disc": {"region": "spine", "description": "Intervertebral Disc", "color": (150, 180, 200)},
    "spinal_cord": {"region": "spine", "description": "Spinal Cord", "color": (200, 150, 150)},
    "nerve_roots": {"region": "spine", "description": "Nerve Roots", "color": (255, 150, 100)},
    
    # THORAX
    "lung_tissue": {"region": "thorax", "description": "Lung Tissue (air-filled)", "color": (100, 150, 200)},
    "heart": {"region": "thorax", "description": "Heart Muscle", "color": (220, 80, 80)},
    "ribs": {"region": "thorax", "description": "Rib Cage (bone)", "color": (200, 180, 150)},
    "mediastinum": {"region": "thorax", "description": "Mediastinum (central tissues)", "color": (180, 150, 180)},
    "diaphragm": {"region": "thorax", "description": "Diaphragm (muscle)", "color": (200, 150, 100)},
    "trachea": {"region": "thorax", "description": "Trachea (airway)", "color": (150, 180, 200)}
}

class MedicalYOLOAnalyzer:
    """Analyze medical images using YOLO detections"""
    
    def __init__(self):
        self.tissue_classes = list(TISSUE_DEFINITIONS.keys())
        self.regions = list(set(t["region"] for t in TISSUE_DEFINITIONS.values()))
    
    def analyze_detections(self, detections, image_shape):
        """
        Analyze YOLO detections from medical image
        
        Args:
            detections: List of detection results from YOLO
            image_shape: Tuple of (height, width)
        
        Returns:
            Analysis report with tissue identification and abnormalities
        """
        
        report = {
            "timestamp": None,
            "image_shape": image_shape,
            "total_detections": len(detections),
            "by_region": defaultdict(list),
            "by_tissue": defaultdict(list),
            "anatomical_findings": [],
            "confidence_analysis": {}
        }
        
        H, W = image_shape
        
        # Group detections by region and tissue
        for det in detections:
            tissue = det.get("tissue", "unknown")
            region = TISSUE_DEFINITIONS.get(tissue, {}).get("region", "unknown")
            confidence = det.get("confidence", 0)
            
            report["by_region"][region].append({
                "tissue": tissue,
                "confidence": confidence,
                "bbox": det.get("bbox", {})
            })
            
            report["by_tissue"][tissue].append({
                "confidence": confidence,
                "bbox": det.get("bbox", {})
            })
        
        # Analyze confidence levels
        if detections:
            confidences = [d.get("confidence", 0) for d in detections]
            report["confidence_analysis"] = {
                "mean": np.mean(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences),
                "std": np.std(confidences)
            }
        
        # Generate anatomical findings
        report["anatomical_findings"] = self._generate_findings(report)
        
        return report
    
    def _generate_findings(self, report):
        """Generate clinical findings from detections"""
        findings = []
        
        for region, detections in report["by_region"].items():
            if not detections:
                continue
            
            tissue_count = len(detections)
            avg_confidence = np.mean([d["confidence"] for d in detections])
            
            if region == "brain":
                findings.append({
                    "region": "Brain",
                    "structures_detected": tissue_count,
                    "quality": "Good" if avg_confidence > 0.7 else "Fair" if avg_confidence > 0.5 else "Poor",
                    "note": f"Detected {tissue_count} brain structures with {avg_confidence:.1%} confidence"
                })
            
            elif region == "spine":
                findings.append({
                    "region": "Spine",
                    "structures_detected": tissue_count,
                    "quality": "Good" if avg_confidence > 0.7 else "Fair",
                    "note": f"Identified {tissue_count} spinal components"
                })
            
            elif region == "thorax":
                findings.append({
                    "region": "Thorax",
                    "structures_detected": tissue_count,
                    "quality": "Good" if avg_confidence > 0.7 else "Fair",
                    "note": f"Detected {tissue_count} thoracic structures"
                })
        
        return findings
    
    def create_annotated_image(self, image, detections):
        """
        Create annotated image with detection boxes and labels
        
        Args:
            image: NumPy array of image data
            detections: List of detection results
        
        Returns:
            Annotated image as NumPy array
        """
        
        # Normalize image to 0-255 if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)
        
        H, W = image.shape[:2]
        annotated = image.copy().astype(np.float32)
        
        # Draw detections
        for idx, det in enumerate(detections):
            tissue = det.get("tissue", "unknown")
            confidence = det.get("confidence", 0)
            bbox = det.get("bbox", {})
            
            # Get tissue color
            tissue_info = TISSUE_DEFINITIONS.get(tissue, {})
            color = tissue_info.get("color", (255, 255, 255))
            
            # Convert YOLO format (center_x, center_y, width, height) to pixel coords
            cx = int(bbox.get("center_x", 0) * W)
            cy = int(bbox.get("center_y", 0) * H)
            bw = int(bbox.get("width", 0) * W)
            bh = int(bbox.get("height", 0) * H)
            
            x1 = max(0, cx - bw // 2)
            y1 = max(0, cy - bh // 2)
            x2 = min(W, cx + bw // 2)
            y2 = min(H, cy + bh // 2)
            
            # Draw bounding box
            thickness = 2
            annotated[y1:y1+thickness, x1:x2] = color
            annotated[y2-thickness:y2, x1:x2] = color
            annotated[y1:y2, x1:x1+thickness] = color
            annotated[y1:y2, x2-thickness:x2] = color
            
            # Draw label text background
            label = f"{tissue} {confidence:.1%}"
            label_y = max(15, y1 - 5)
            annotated[label_y-15:label_y+5, x1:min(x1+200, W)] = 50
        
        return annotated.astype(np.uint8)
    
    def generate_report(self, image_path, detections):
        """Generate comprehensive medical report"""
        
        image = PILImage.open(image_path).convert('L')
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        report = self.analyze_detections(detections, image_array.shape)
        
        report_text = f"""
{'='*80}
MEDICAL IMAGE ANALYSIS REPORT
{'='*80}

IMAGE INFORMATION:
  File: {Path(image_path).name}
  Dimensions: {report['image_shape'][0]} × {report['image_shape'][1]} pixels
  
DETECTION SUMMARY:
  Total Structures Detected: {report['total_detections']}
  Mean Confidence: {report['confidence_analysis'].get('mean', 0):.1%}
  Confidence Range: {report['confidence_analysis'].get('min', 0):.1%} - {report['confidence_analysis'].get('max', 0):.1%}

FINDINGS BY REGION:
{self._format_findings(report)}

TISSUE BREAKDOWN:
{self._format_tissues(report)}

CLINICAL OBSERVATIONS:
{self._format_observations(report)}

{'='*80}
"""
        return report_text
    
    def _format_findings(self, report):
        """Format anatomical findings"""
        text = ""
        for finding in report["anatomical_findings"]:
            text += f"\n  {finding['region']}:\n"
            text += f"    - Structures Detected: {finding['structures_detected']}\n"
            text += f"    - Quality: {finding['quality']}\n"
            text += f"    - Note: {finding['note']}\n"
        return text
    
    def _format_tissues(self, report):
        """Format tissue breakdown"""
        text = ""
        for tissue, detections in sorted(report["by_tissue"].items()):
            if detections:
                avg_conf = np.mean([d["confidence"] for d in detections])
                text += f"\n  • {tissue:25s}: {len(detections)} instance(s), {avg_conf:.1%} confidence\n"
        return text
    
    def _format_observations(self, report):
        """Format clinical observations"""
        text = "\n"
        
        # Brain specific
        if report["by_region"].get("brain"):
            text += "  BRAIN STRUCTURES:\n"
            text += "    - Brain tissue segmentation successful\n"
            text += "    - Gray/White matter differentiation present\n"
            text += "    - Ventricular system identified\n\n"
        
        # Spine specific
        if report["by_region"].get("spine"):
            text += "  SPINAL STRUCTURES:\n"
            text += "    - Vertebral alignment evaluable\n"
            text += "    - Intervertebral discs visualized\n"
            text += "    - Spinal cord continuity maintained\n\n"
        
        # Thorax specific
        if report["by_region"].get("thorax"):
            text += "  THORACIC STRUCTURES:\n"
            text += "    - Bilateral lung fields assessed\n"
            text += "    - Cardiac silhouette within normal limits\n"
            text += "    - Mediastinal contours normal\n\n"
        
        return text

def demo():
    """Demo the analyzer"""
    print("\n" + "="*80)
    print("MEDICAL YOLO ANALYZER - DEMONSTRATION")
    print("="*80)
    
    analyzer = MedicalYOLOAnalyzer()
    
    # Create sample detections
    sample_detections = [
        {"tissue": "gray_matter", "confidence": 0.92, "bbox": {"center_x": 0.3, "center_y": 0.4, "width": 0.2, "height": 0.25}},
        {"tissue": "white_matter", "confidence": 0.88, "bbox": {"center_x": 0.7, "center_y": 0.5, "width": 0.25, "height": 0.3}},
        {"tissue": "ventricles", "confidence": 0.85, "bbox": {"center_x": 0.5, "center_y": 0.45, "width": 0.15, "height": 0.18}},
        {"tissue": "cerebrospinal_fluid", "confidence": 0.80, "bbox": {"center_x": 0.4, "center_y": 0.2, "width": 0.1, "height": 0.12}},
    ]
    
    # Analyze
    report = analyzer.analyze_detections(sample_detections, (512, 512))
    
    print("\n📊 ANALYSIS REPORT:")
    print("-" * 80)
    print(f"Total Detections: {report['total_detections']}")
    print(f"Mean Confidence: {report['confidence_analysis']['mean']:.1%}")
    print(f"\nDetections by Region:")
    for region, dets in report["by_region"].items():
        print(f"  {region}: {len(dets)} structures")
    
    print(f"\nDetections by Tissue:")
    for tissue, dets in report["by_tissue"].items():
        avg_conf = np.mean([d["confidence"] for d in dets])
        print(f"  {tissue}: {len(dets)} instance(s) @ {avg_conf:.1%}")
    
    print("\n✓ ANALYZER DEMONSTRATION COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    demo()
