#!/usr/bin/env python3
"""
YOLO Training Script for Medical Image Detection
Trains YOLO on synthetic medical data with different tissue types:
- Brain tissues (gray matter, white matter, ventricles)
- Spine components (vertebrae, discs, spinal cord)
- Thorax regions (lungs, heart, ribs, mediastinum)

Run: python train_yolo_medical.py
"""

import numpy as np
from pathlib import Path
import json
import tempfile
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from datetime import datetime

print("\n" + "="*80)
print("YOLO MEDICAL TISSUE TRAINING SYSTEM")
print("="*80)

# Define tissue types and their characteristics
TISSUE_DEFINITIONS = {
    # BRAIN TISSUES
    "gray_matter": {
        "region": "brain",
        "intensity_range": (0.4, 0.7),
        "description": "Gray Matter (neurons)",
        "color": (80, 120, 200),
        "typical_size": (20, 60)
    },
    "white_matter": {
        "region": "brain",
        "intensity_range": (0.6, 0.85),
        "description": "White Matter (axons)",
        "color": (200, 200, 220),
        "typical_size": (30, 80)
    },
    "ventricles": {
        "region": "brain",
        "intensity_range": (0.9, 1.0),
        "description": "Brain Ventricles (fluid)",
        "color": (50, 100, 150),
        "typical_size": (15, 40)
    },
    "cerebrospinal_fluid": {
        "region": "brain",
        "intensity_range": (0.85, 0.95),
        "description": "Cerebrospinal Fluid",
        "color": (100, 150, 200),
        "typical_size": (10, 30)
    },
    
    # SPINE COMPONENTS
    "vertebral_body": {
        "region": "spine",
        "intensity_range": (0.5, 0.75),
        "description": "Vertebral Body (bone)",
        "color": (220, 200, 150),
        "typical_size": (40, 100)
    },
    "intervertebral_disc": {
        "region": "spine",
        "intensity_range": (0.35, 0.55),
        "description": "Intervertebral Disc",
        "color": (150, 180, 200),
        "typical_size": (30, 60)
    },
    "spinal_cord": {
        "region": "spine",
        "intensity_range": (0.4, 0.65),
        "description": "Spinal Cord",
        "color": (200, 150, 150),
        "typical_size": (10, 25)
    },
    "nerve_roots": {
        "region": "spine",
        "intensity_range": (0.3, 0.5),
        "description": "Nerve Roots",
        "color": (255, 150, 100),
        "typical_size": (5, 15)
    },
    
    # THORAX REGIONS
    "lung_tissue": {
        "region": "thorax",
        "intensity_range": (0.2, 0.4),
        "description": "Lung Tissue (air-filled)",
        "color": (100, 150, 200),
        "typical_size": (80, 200)
    },
    "heart": {
        "region": "thorax",
        "intensity_range": (0.5, 0.75),
        "description": "Heart Muscle",
        "color": (220, 80, 80),
        "typical_size": (60, 120)
    },
    "ribs": {
        "region": "thorax",
        "intensity_range": (0.65, 0.85),
        "description": "Rib Cage (bone)",
        "color": (200, 180, 150),
        "typical_size": (40, 100)
    },
    "mediastinum": {
        "region": "thorax",
        "intensity_range": (0.45, 0.65),
        "description": "Mediastinum (central tissues)",
        "color": (180, 150, 180),
        "typical_size": (50, 120)
    },
    "diaphragm": {
        "region": "thorax",
        "intensity_range": (0.55, 0.70),
        "description": "Diaphragm (muscle)",
        "color": (200, 150, 100),
        "typical_size": (30, 80)
    },
    "trachea": {
        "region": "thorax",
        "intensity_range": (0.3, 0.5),
        "description": "Trachea (airway)",
        "color": (150, 180, 200),
        "typical_size": (10, 30)
    }
}

class SyntheticMedicalImageGenerator:
    """Generate synthetic medical images with annotations"""
    
    def __init__(self, image_size=(512, 512)):
        self.image_size = image_size
        self.H, self.W = image_size
    
    def create_background(self, region_type="brain"):
        """Create anatomical background based on region"""
        bg = np.ones((self.H, self.W), dtype=np.float32) * 0.2
        
        # Add anatomical structure
        y, x = np.ogrid[:self.H, :self.W]
        
        if region_type == "brain":
            # Simulate brain oval shape
            mask = ((y - self.H//2)**2 / (self.H//2.5)**2 + 
                   (x - self.W//2)**2 / (self.W//2.3)**2 < 1)
            bg[mask] = 0.35
        elif region_type == "spine":
            # Vertical structure
            spine_mask = np.abs(x - self.W//2) < 30
            bg[spine_mask] = 0.4
        elif region_type == "thorax":
            # Elliptical thorax shape
            mask = ((y - self.H//2)**2 / (self.H//2.2)**2 + 
                   (x - self.W//2)**2 / (self.W//2.5)**2 < 1)
            bg[mask] = 0.35
        
        return bg
    
    def add_tissue(self, image, tissue_type, num_instances=None):
        """Add tissue structure to image"""
        if tissue_type not in TISSUE_DEFINITIONS:
            return image, []
        
        tissue_info = TISSUE_DEFINITIONS[tissue_type]
        if num_instances is None:
            num_instances = np.random.randint(1, 4)
        
        detections = []
        intensity_min, intensity_max = tissue_info["intensity_range"]
        size_min, size_max = tissue_info["typical_size"]
        
        for _ in range(num_instances):
            # Random position
            h = np.random.randint(size_min, size_max)
            w = np.random.randint(size_min, size_max)
            y = np.random.randint(h, self.H - h)
            x = np.random.randint(w, self.W - w)
            
            # Create tissue blob with Gaussian profile
            yy, xx = np.ogrid[-h//2:h//2, -w//2:w//2]
            mask = np.exp(-(yy**2 + xx**2) / (2 * (h/4)**2))
            
            # Add intensity variation
            intensity = np.random.uniform(intensity_min, intensity_max)
            image[y-h//2:y+h//2, x-w//2:x+w//2] = np.maximum(
                image[y-h//2:y+h//2, x-w//2:x+w//2],
                mask * intensity
            )
            
            # Normalize bounding box (YOLO format: center_x, center_y, width, height)
            detections.append({
                "tissue": tissue_type,
                "bbox": {
                    "center_x": x / self.W,
                    "center_y": y / self.H,
                    "width": w / self.W,
                    "height": h / self.H
                },
                "confidence": intensity,
                "region": tissue_info["region"],
                "intensity_range": tissue_info["intensity_range"]
            })
        
        return image, detections
    
    def generate_brain_scan(self):
        """Generate synthetic brain MRI"""
        image = self.create_background("brain")
        all_detections = []
        
        brain_tissues = ["gray_matter", "white_matter", "ventricles", "cerebrospinal_fluid"]
        for tissue in brain_tissues:
            image, dets = self.add_tissue(image, tissue, num_instances=np.random.randint(2, 4))
            all_detections.extend(dets)
        
        return image, all_detections
    
    def generate_spine_scan(self):
        """Generate synthetic spine MRI"""
        image = self.create_background("spine")
        all_detections = []
        
        spine_tissues = ["vertebral_body", "intervertebral_disc", "spinal_cord", "nerve_roots"]
        # Stack vertebrae
        num_vertebrae = 5
        for v in range(num_vertebrae):
            y_offset = (v + 1) * (self.H // (num_vertebrae + 1))
            for tissue in spine_tissues:
                image_local, dets = self.add_tissue(image, tissue, num_instances=1)
                # Adjust detection coordinates for vertical offset
                for det in dets:
                    det["bbox"]["center_y"] += y_offset / self.H
                    all_detections.append(det)
        
        return image, all_detections
    
    def generate_thorax_scan(self):
        """Generate synthetic thorax CT/X-ray"""
        image = self.create_background("thorax")
        all_detections = []
        
        thorax_tissues = ["lung_tissue", "heart", "ribs", "mediastinum", "diaphragm", "trachea"]
        for tissue in thorax_tissues:
            num = 2 if tissue in ["lung_tissue", "ribs"] else 1
            image, dets = self.add_tissue(image, tissue, num_instances=num)
            all_detections.extend(dets)
        
        return image, all_detections

class YOLOMedicalTrainer:
    """Train YOLO model on medical data"""
    
    def __init__(self, output_dir="medical_yolo_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.generator = SyntheticMedicalImageGenerator()
        self.dataset_config = {
            "name": "Medical Tissue Detection Dataset",
            "classes": list(TISSUE_DEFINITIONS.keys()),
            "regions": list(set(t["region"] for t in TISSUE_DEFINITIONS.values())),
            "created": datetime.now().isoformat(),
            "statistics": {}
        }
    
    def generate_training_data(self, num_images_per_type=50):
        """Generate training dataset"""
        print("\n📊 GENERATING TRAINING DATA")
        print("-" * 80)
        
        split_dirs = {
            "images": self.output_dir / "images",
            "labels": self.output_dir / "labels"
        }
        
        for dir_path in split_dirs.values():
            dir_path.mkdir(exist_ok=True, parents=True)
        
        image_count = 0
        annotations_list = []
        region_counts = {"brain": 0, "spine": 0, "thorax": 0}
        
        generators = {
            "brain": self.generator.generate_brain_scan,
            "spine": self.generator.generate_spine_scan,
            "thorax": self.generator.generate_thorax_scan
        }
        
        for region_name, gen_func in generators.items():
            print(f"\n  📈 Generating {num_images_per_type} {region_name.upper()} scans...")
            
            for i in range(num_images_per_type):
                # Generate synthetic image
                image, detections = gen_func()
                
                # Save image
                image_normalized = (np.clip(image, 0, 1) * 255).astype(np.uint8)
                img_pil = PILImage.fromarray(image_normalized)
                img_path = split_dirs["images"] / f"{region_name}_{image_count:04d}.png"
                img_pil.save(img_path)
                
                # Save YOLO format annotations
                label_path = split_dirs["labels"] / f"{region_name}_{image_count:04d}.txt"
                with open(label_path, 'w') as f:
                    for det in detections:
                        tissue = det["tissue"]
                        class_id = list(TISSUE_DEFINITIONS.keys()).index(tissue)
                        bbox = det["bbox"]
                        f.write(f"{class_id} {bbox['center_x']:.4f} {bbox['center_y']:.4f} "
                               f"{bbox['width']:.4f} {bbox['height']:.4f}\n")
                
                annotations_list.append({
                    "image": str(img_path),
                    "region": region_name,
                    "detections": detections
                })
                
                region_counts[region_name] += 1
                image_count += 1
                
                if (i + 1) % 10 == 0:
                    print(f"    ✓ {i + 1}/{num_images_per_type} {region_name} images")
        
        self.dataset_config["statistics"] = {
            "total_images": image_count,
            "region_distribution": region_counts,
            "total_tissue_detections": sum(len(ann["detections"]) for ann in annotations_list)
        }
        
        return annotations_list
    
    def create_dataset_yaml(self):
        """Create YOLO dataset configuration"""
        dataset_yaml = {
            "path": str(self.output_dir),
            "train": "images",
            "val": "images",
            "test": "images",
            "nc": len(TISSUE_DEFINITIONS),
            "names": {i: tissue for i, tissue in enumerate(TISSUE_DEFINITIONS.keys())},
            "tissue_metadata": {
                tissue: {
                    "region": info["region"],
                    "description": info["description"],
                    "intensity_range": info["intensity_range"],
                    "typical_size": info["typical_size"]
                }
                for tissue, info in TISSUE_DEFINITIONS.items()
            }
        }
        
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            json.dump(dataset_yaml, f, indent=2)
        
        return dataset_yaml
    
    def train(self, num_images_per_type=50, epochs=50):
        """Complete training pipeline"""
        print("\n" + "="*80)
        print("YOLO MEDICAL TISSUE DETECTOR - TRAINING")
        print("="*80)
        
        # 1. Generate data
        print("\n[1/4] Generating synthetic medical images...")
        annotations = self.generate_training_data(num_images_per_type)
        
        # 2. Create dataset config
        print("\n[2/4] Creating dataset configuration...")
        dataset_config = self.create_dataset_yaml()
        
        # 3. Print statistics
        print("\n[3/4] Dataset Statistics:")
        print("-" * 80)
        stats = self.dataset_config["statistics"]
        print(f"  • Total Images: {stats['total_images']}")
        print(f"  • Brain Scans: {stats['region_distribution']['brain']}")
        print(f"  • Spine Scans: {stats['region_distribution']['spine']}")
        print(f"  • Thorax Scans: {stats['region_distribution']['thorax']}")
        print(f"  • Total Tissue Detections: {stats['total_tissue_detections']}")
        print(f"  • Tissue Classes: {len(TISSUE_DEFINITIONS)}")
        
        # 4. Print tissue information
        print("\n[4/4] Tissue Classes by Region:")
        print("-" * 80)
        
        for region in ["brain", "spine", "thorax"]:
            print(f"\n  {region.upper()}:")
            region_tissues = [t for t, info in TISSUE_DEFINITIONS.items() 
                            if info["region"] == region]
            for tissue in region_tissues:
                info = TISSUE_DEFINITIONS[tissue]
                print(f"    • {tissue:25s} → {info['description']}")
                print(f"      Intensity: {info['intensity_range']}, Size: {info['typical_size']} px")
        
        # 5. Training command
        print("\n" + "="*80)
        print("NEXT STEPS - TRAINING COMMAND:")
        print("="*80)
        print(f"""
To train YOLOv8 on this dataset, use:

    from ultralytics import YOLO
    
    model = YOLO('yolov8n.pt')
    results = model.train(
        data='{self.output_dir}/data.yaml',
        epochs={epochs},
        imgsz=512,
        batch=16,
        patience=20,
        save=True,
        device=0
    )
    
    # Test the model
    results = model.predict(source='path/to/medical/image.png')
    model.export(format='onnx')  # Export for deployment

Dataset Location: {self.output_dir}
Images: {self.output_dir / 'images'}
Labels: {self.output_dir / 'labels'}
Config: {self.output_dir / 'data.yaml'}
""")
        
        print("="*80)
        print("✓ TRAINING DATA GENERATION COMPLETE")
        print("="*80 + "\n")
        
        return annotations, dataset_config

def main():
    # Initialize trainer
    trainer = YOLOMedicalTrainer(output_dir="medical_yolo_dataset")
    
    # Generate training data
    annotations, config = trainer.train(
        num_images_per_type=50,  # 150 total images (50 per region)
        epochs=50
    )
    
    # Additional information
    print("\nDATASET INFORMATION:")
    print(f"  Dataset Path: {trainer.output_dir}")
    print(f"  Classes: {config['nc']}")
    print(f"  Total Images: {config['statistics']['total_images']}")
    print(f"\nTissue Classes ({len(TISSUE_DEFINITIONS)}):")
    for i, (tissue, info) in enumerate(TISSUE_DEFINITIONS.items()):
        print(f"  {i:2d}. {tissue:25s} ({info['region']:6s}) → {info['description']}")

if __name__ == "__main__":
    main()
