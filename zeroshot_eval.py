import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModel, AutoProcessor
from datasets import load_dataset
import pandas as pd
import numpy as np

# Configuration
MODEL_NAME = "google/siglip2-base-patch16-256"
OUTPUT_DIR = "/home/baha4001/hyeonjun/siglip_analysis/results_paper_recipe"
BATCH_SIZE = 32  # Smaller batch size due to processing 1000 texts per image
NUM_GPUS = torch.cuda.device_count()


def get_imagenet_classes():
    """Get ImageNet class names from HuggingFace dataset."""
    print("Loading ImageNet class names from HuggingFace...")

    # Load dataset info to get class names
    ds_builder = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

    # Get features info
    features = ds_builder.features
    class_names = features["label"].names

    # Use only the first synonym (before comma) for each class
    # Apply template and lowercase as per paper recipe
    clean_names = []
    for name in class_names:
        # Take first part before comma (first synonym)
        first_name = name.split(",")[0].strip()
        # Apply lowercase and template (paper recipe)
        clean_names.append(f"this is a photo of {first_name.lower()}.")

    print(f"Loaded {len(clean_names)} class names")
    return clean_names


def process_single_image(model, processor, image, class_names, device):
    """Process a single image using official SigLIP zero-shot method."""
    # Get base model if using DataParallel
    base_model = model.module if hasattr(model, 'module') else model

    # Prepare inputs: image with all class texts
    # Paper recipe: padding="max_length", max_length=64
    inputs = processor(
        text=class_names,
        images=image,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = base_model(**inputs)
        # logits_per_image already includes logit_scale and logit_bias
        logits = outputs.logits_per_image[0]  # (num_classes,)
        probs = torch.sigmoid(logits)

        # Get top-5 predictions
        top5_probs, top5_indices = probs.topk(5)

    return top5_indices.cpu().numpy(), top5_probs.cpu().numpy()


def run_evaluation():
    """Run zero-shot evaluation on ImageNet validation set."""
    print("=" * 60)
    print("SigLIP Zero-shot Classification Evaluation (Fixed)")
    print("=" * 60)
    print(f"Available GPUs: {NUM_GPUS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class names from HuggingFace
    class_names = get_imagenet_classes()
    assert len(class_names) == 1000, f"Expected 1000 classes, got {len(class_names)}"

    # Load model and processor
    print(f"\nLoading model: {MODEL_NAME}")
    model = AutoModel.from_pretrained(MODEL_NAME).eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Move to GPU with DataParallel (4 GPUs)
    if NUM_GPUS > 1:
        print(f"Using DataParallel with {NUM_GPUS} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    print(f"Model loaded on {device}")

    # Load ImageNet validation set (streaming mode)
    print("\nLoading ImageNet-1K validation set (streaming)...")
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

    # Initialize results storage
    results = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_top5_correct = defaultdict(int)
    confusion_matrix = defaultdict(lambda: defaultdict(int))

    total_correct = 0
    total_top5_correct = 0
    total_processed = 0

    print(f"\nProcessing images one by one (official SigLIP method)...")
    pbar = tqdm(enumerate(dataset), total=50000, desc="Evaluating")

    for idx, sample in pbar:
        image = sample["image"]
        true_label = sample["label"]

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process single image with all class texts
        top5_preds, top5_probs = process_single_image(model, processor, image, class_names, device)

        pred_label = int(top5_preds[0])
        correct = (pred_label == true_label)
        top5_correct = (true_label in top5_preds)

        # Update counters
        total_correct += correct
        total_top5_correct += top5_correct
        class_correct[true_label] += correct
        class_top5_correct[true_label] += top5_correct
        class_total[true_label] += 1

        # Track confusion
        if not correct:
            confusion_matrix[true_label][pred_label] += 1

        # Store result
        results.append({
            "image_idx": idx,
            "true_class": true_label,
            "true_label": class_names[true_label],
            "pred_class": pred_label,
            "pred_label": class_names[pred_label],
            "correct": int(correct),
            "top5_correct": int(top5_correct),
            "top5_preds": ",".join([str(p) for p in top5_preds])
        })

        total_processed += 1

        # Update progress bar
        if total_processed > 0:
            top1_acc = total_correct / total_processed * 100
            top5_acc = total_top5_correct / total_processed * 100
            pbar.set_postfix({"Top1": f"{top1_acc:.2f}%", "Top5": f"{top5_acc:.2f}%"})

    # Calculate final metrics
    top1_accuracy = total_correct / total_processed * 100
    top5_accuracy = total_top5_correct / total_processed * 100

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total images processed: {total_processed}")
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")

    # Save results
    save_results(results, class_names, class_correct, class_top5_correct,
                 class_total, confusion_matrix, top1_accuracy, top5_accuracy)


def save_results(results, class_names, class_correct, class_top5_correct,
                 class_total, confusion_matrix, top1_accuracy, top5_accuracy):
    """Save all results to files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Save predictions (all 50,000 images)
    print("\nSaving predictions.csv...")
    df_predictions = pd.DataFrame(results)
    df_predictions.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
    print(f"  Saved {len(results)} predictions")

    # 2. Save class accuracy (all 1000 classes)
    print("Saving class_accuracy.csv...")
    class_acc_data = []
    for class_idx in range(len(class_names)):
        total = class_total.get(class_idx, 0)
        correct = class_correct.get(class_idx, 0)
        top5_correct = class_top5_correct.get(class_idx, 0)

        acc = correct / total * 100 if total > 0 else 0
        top5_acc = top5_correct / total * 100 if total > 0 else 0

        class_acc_data.append({
            "class_idx": class_idx,
            "class_name": class_names[class_idx],
            "total": total,
            "correct": correct,
            "top1_accuracy": acc,
            "top5_correct": top5_correct,
            "top5_accuracy": top5_acc
        })

    df_class_acc = pd.DataFrame(class_acc_data)
    df_class_acc.to_csv(os.path.join(OUTPUT_DIR, "class_accuracy.csv"), index=False)

    # 3. Save worst classes (Top 50 lowest accuracy)
    print("Saving worst_classes.csv...")
    df_worst = df_class_acc.sort_values("top1_accuracy").head(50)
    df_worst.to_csv(os.path.join(OUTPUT_DIR, "worst_classes.csv"), index=False)

    print("\n" + "-" * 50)
    print("Top 50 Worst Classes (Lowest Top-1 Accuracy):")
    print("-" * 50)
    for _, row in df_worst.head(20).iterrows():
        print(f"{row['class_idx']:4d} {row['class_name']:35s} : {row['top1_accuracy']:5.1f}% ({row['correct']:2d}/{row['total']:2d})")

    # 4. Save confusion pairs (all misclassifications)
    print("\nSaving confusion_pairs.csv...")
    confusion_pairs = []
    for true_class, pred_dict in confusion_matrix.items():
        for pred_class, count in pred_dict.items():
            confusion_pairs.append({
                "true_class": int(true_class),
                "true_label": class_names[true_class],
                "pred_class": int(pred_class),
                "pred_label": class_names[pred_class],
                "count": count
            })

    df_confusion = pd.DataFrame(confusion_pairs)
    df_confusion = df_confusion.sort_values("count", ascending=False)
    df_confusion.to_csv(os.path.join(OUTPUT_DIR, "confusion_pairs.csv"), index=False)

    print("\n" + "-" * 50)
    print("Top 20 Confusion Pairs (Most Frequent Misclassifications):")
    print("-" * 50)
    for _, row in df_confusion.head(20).iterrows():
        print(f"{row['true_label']:30s} -> {row['pred_label']:30s} : {row['count']}")

    # 5. Save summary
    print("\nSaving summary.json...")
    summary = {
        "model": MODEL_NAME,
        "dataset": "ImageNet-1K Validation",
        "total_images": len(results),
        "top1_accuracy": round(top1_accuracy, 4),
        "top5_accuracy": round(top5_accuracy, 4),
        "num_classes": len(class_names),
        "method": "official SigLIP (logits_per_image with sigmoid)",
        "text_prompt": "this is a photo of {class_name}. (lowercase, max_length=64)"
    }

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 6. Save detailed class statistics
    print("Saving class_statistics.json...")
    class_stats = {
        "classes_with_0_accuracy": df_class_acc[df_class_acc["top1_accuracy"] == 0]["class_name"].tolist(),
        "classes_with_100_accuracy": df_class_acc[df_class_acc["top1_accuracy"] == 100]["class_name"].tolist(),
        "mean_class_accuracy": df_class_acc["top1_accuracy"].mean(),
        "median_class_accuracy": df_class_acc["top1_accuracy"].median(),
        "std_class_accuracy": df_class_acc["top1_accuracy"].std()
    }

    with open(os.path.join(OUTPUT_DIR, "class_statistics.json"), "w") as f:
        json.dump(class_stats, f, indent=2)

    print("\n" + "=" * 60)
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 60)
    print("\nOutput files:")
    print(f"  - predictions.csv       : {len(results)} image predictions")
    print(f"  - class_accuracy.csv    : {len(class_names)} class accuracies")
    print(f"  - worst_classes.csv     : Top 50 worst performing classes")
    print(f"  - confusion_pairs.csv   : {len(confusion_pairs)} confusion pairs")
    print(f"  - summary.json          : Overall metrics")
    print(f"  - class_statistics.json : Detailed statistics")


if __name__ == "__main__":
    run_evaluation()
