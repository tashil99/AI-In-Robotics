# scripts/training-model.py
import os
import pickle
from ultralytics import YOLO


def analyze_weak_classes(model_path):
    """Analyze which classes are underperforming"""
    model = YOLO(model_path)

    # Get validation results
    metrics = model.val()

    # Print class-wise metrics
    print("\n=== CLASS PERFORMANCE ANALYSIS ===")
    for i, class_name in enumerate(metrics.names):
        precision = metrics.box.mp[i] if hasattr(metrics.box, 'mp') else 0
        recall = metrics.box.mr[i] if hasattr(metrics.box, 'mr') else 0
        print(f"{class_name}: Precision={precision:.3f}, Recall={recall:.3f}")

    # Identify weak classes (thresholds can be adjusted)
    weak_classes = []
    for i, class_name in enumerate(metrics.names):
        precision = metrics.box.mp[i] if hasattr(metrics.box, 'mp') else 0
        recall = metrics.box.mr[i] if hasattr(metrics.box, 'mr') else 0

        if precision < 0.7 or recall < 0.6:  # Adjust thresholds as needed
            weak_classes.append(i)
            print(f"🚨 WEAK CLASS DETECTED: {class_name}")

    return weak_classes, metrics.names


def main():
    # Avoid OpenMP duplicate runtime crash
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # CPU-specific performance tweaks
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["ULTRALYTICS_CACHE"] = "ram"

    # Step 1: Analyze current model to identify weak classes
    model_path = "runs/detect/AI-In-Robotics-CPU-Exp502/weights/best.pt"
    weak_classes, class_names = analyze_weak_classes(model_path)

    print(f"\n🎯 Target weak classes: {[class_names[i] for i in weak_classes]}")

    # Load model for fine-tuning
    model = YOLO(model_path)

    # Enhanced training configuration for weak class improvement
    results = model.train(
        data="../dataset/data.yaml",
        epochs=30,  # Shorter for fine-tuning
        imgsz=512,
        batch=8,
        name="AI-In-Robotics-CPU-Exp503-Finetune",  # New experiment name
        verbose=True,

        # Optimizer & Learning Rate
        optimizer="AdamW",
        lr0=0.0001,  # Lower learning rate for fine-tuning
        lrf=0.01,  # Final learning rate

        # Data Augmentation - Enhanced for weak classes
        degrees=10.0,  # Increased rotation
        translate=0.2,  # Increased translation
        scale=0.3,  # Increased scaling
        shear=5.0,  # Increased shear
        perspective=0.001,  # Add perspective transform
        flipud=0.1,  # Flip up-down
        fliplr=0.5,  # Flip left-right

        # Augmentation strengths
        mosaic=0.8,  # Keep mosaic but not too high
        mixup=0.1,  # Reduced mixup to avoid confusion
        copy_paste=0.1,  # Add copy-paste augmentation

        # Color augmentations
        hsv_h=0.02,  # Increased hue variation
        hsv_s=0.7,  # Increased saturation
        hsv_v=0.4,  # Increased value

        # Training strategy
        patience=15,  # Early stopping patience
        cos_lr=True,  # Cosine learning rate scheduler
        warmup_epochs=3,  # Learning rate warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Class balancing (if supported)
        # class_weights=[1.0, 1.0, 1.0, 2.0, 2.0, 1.0]  # Adjust based on weak_classes

        # Overlap and NMS settings
        overlap_mask=True,
        mask_ratio=2,

        # Validation
        val=True,
        save_period=5,  # Save checkpoint every 5 epochs
    )

    # Save results
    results_dir = "runs/detect/AI-In-Robotics-CPU-Exp503-Finetune"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    # Generate performance comparison
    generate_performance_comparison(model_path, results_dir, class_names)

    print(f"\n✅ Fine-tuning complete! Results saved to: {results_path}")
    print(f"📂 Plots, labels, and checkpoints are in '{results_dir}'")


def generate_performance_comparison(original_model_path, new_results_dir, class_names):
    """Compare performance before and after fine-tuning"""
    original_model = YOLO(original_model_path)
    new_model = YOLO(os.path.join(new_results_dir, "weights", "best.pt"))

    print("\n📊 PERFORMANCE COMPARISON")
    print("Class\t\tOriginal mAP50\tNew mAP50\tImprovement")
    print("-" * 60)

    # This is a simplified comparison - you might need to run proper validation
    orig_metrics = original_model.val()
    new_metrics = new_model.val()

    for i, class_name in enumerate(class_names):
        orig_map = getattr(orig_metrics.box, 'map50', 0) if hasattr(orig_metrics.box, 'map50') else 0
        new_map = getattr(new_metrics.box, 'map50', 0) if hasattr(new_metrics.box, 'map50') else 0
        improvement = new_map - orig_map

        print(f"{class_name:12}\t{orig_map:.3f}\t\t{new_map:.3f}\t\t{improvement:+.3f}")


def create_focused_augmentation_pipeline():
    """Create augmentation pipeline specifically for weak classes"""
    # This would be used if you're doing custom data augmentation
    # outside of YOLO's built-in augmentations
    pass


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()