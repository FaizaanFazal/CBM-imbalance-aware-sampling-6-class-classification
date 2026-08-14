"""Run official scratch 3D DenseNet-121 and EfficientNet-B0 on ImageID Fold 1.

The two executed notebooks are launched strictly sequentially on physical GPU 2.
The fixed test cohort is used only as an ImageID membership set for leakage
auditing; its labels, paths, volumes, predictions, and metrics remain sealed.
"""
from __future__ import annotations

import os

GPU2_UUID = "GPU-db657392-ab6d-94b9-02fd-9db3a5e31c3d"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", GPU2_UUID)
os.environ.setdefault("PYTHONHASHSEED", "42")

import contextlib
import gc
import hashlib
import json
import random
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import monai
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121, EfficientNetBN
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset, RandomSampler

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path("/media/faizaan/4TB/1_DATA_PROJECTS/Projects/Alheimers 6 class classification CBM")
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"
BASELINE_NOTEBOOK_DIR = NOTEBOOK_ROOT / "Baselines"
RESULTS_ROOT = PROJECT_ROOT / "Results"
BASELINE_RESULTS_ROOT = RESULTS_ROOT / "Baselines"
FOLD_DIR = Path("/home/faizaan/AD/projects/HandlingImbalance/folds/fold_1")
REFERENCE_RESULTS = RESULTS_ROOT / "3DResNet_ImageID_Baseline_Scratch"
REFERENCE_NOTEBOOK = NOTEBOOK_ROOT / "3DResNet_ImageID_Baseline_Scratch.ipynb"
STAGE1_NOTEBOOK = NOTEBOOK_ROOT / "stage1_v2.ipynb"

EXPECTED_REFERENCE_NOTEBOOK_SHA256 = "782803ed5b0ed5be1e50ad9af019e075abc518a8bded5e2ba67160aa69324310"
EXPECTED_STAGE1_SHA256 = "b6070971b18cb9f9f2d44821b406b1d755ddd59562af79cfe95be4ebafb815a8"
EXPECTED_TRAIN_SHA256 = "23c2c029b673293bbd313c79eaf220efdd794bf7843118a664f1f8c5bbb038c9"
EXPECTED_VAL_SHA256 = "0b964df6ec6747aa62e6ce2b631c4e439697512f96ba6d1c824bb6e0eadfa5a6"
EXPECTED_TEST_SHA256 = "c58d88969c828ac806a6559798841effce482fd6081b5fa56bce5c91c66fc268"

SEED = 42
NUM_CLASSES = 6
LABEL_MAP = {"AD": 0, "CN": 1, "EMCI": 2, "LMCI": 3, "MCI": 4, "SMC": 5}
CLASS_NAMES = [name for name, index in sorted(LABEL_MAP.items(), key=lambda item: item[1])]
PRETRAINED = False
TEST_INFERENCE_PERFORMED = False
BATCH_SIZE = 4
EFFECTIVE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
NUM_WORKERS = 2
PREFETCH_FACTOR = 1
PIN_MEMORY = True
PERSISTENT_WORKERS = NUM_WORKERS > 0
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 10
MIN_DELTA = 1e-4
INITIAL_LR = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
PLOT_DPI = 600
GPU_IDLE_THRESHOLD_MIB = 256
GPU_IDLE_TIMEOUT_SECONDS = 180


def architecture_spec(key):
    if key == "D":
        return {
            "key": "D",
            "model_name": "3D DenseNet-121",
            "architecture": "Official MONAI DenseNet121, spatial_dims=3, in_channels=1, out_channels=6",
            "results": BASELINE_RESULTS_ROOT / "3DDenseNet_ImageID_Fold1",
            "notebook": BASELINE_NOTEBOOK_DIR / "3DDenseNet_ImageID_Baseline_Fold1.ipynb",
            "decision_prefix": "DENSENET",
        }
    if key == "E":
        return {
            "key": "E",
            "model_name": "3D EfficientNet-B0",
            "architecture": "Official MONAI EfficientNetBN efficientnet-b0, spatial_dims=3, in_channels=1, num_classes=6",
            "results": BASELINE_RESULTS_ROOT / "3DEfficientNet_ImageID_Fold1",
            "notebook": BASELINE_NOTEBOOK_DIR / "3DEfficientNet_ImageID_Baseline_Fold1.ipynb",
            "decision_prefix": "EFFICIENTNET",
        }
    raise ValueError(f"Unknown architecture key: {key}")


ARCHITECTURES = {key: architecture_spec(key) for key in ("D", "E")}
EXECUTION_ORDER = ["D", "E"]
REQUIRED_OUTPUTS = [
    "best_model.pt",
    "history.csv",
    "validation_predictions.csv",
    "validation_metrics.csv",
    "classification_report.csv",
    "per_class_metrics.csv",
    "fold1_split_audit.csv",
    "model_complexity.csv",
    "runtime_summary.json",
    "config.json",
    "health_checks.json",
    "integrity_complete.json",
    "confusion_matrix.png",
    "normalized_confusion_matrix.png",
    "roc_curves.png",
    "learning_curves.png",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path, payload):
    def safe(value):
        if isinstance(value, dict):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    Path(path).write_text(json.dumps(safe(payload), indent=2, allow_nan=False), encoding="utf-8")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def snapshot_preexisting_root_files():
    snapshot = {}
    for path in sorted(NOTEBOOK_ROOT.iterdir()):
        if path.is_file():
            snapshot[path.name] = sha256_file(path)
    return snapshot


def audit_frozen_references():
    audit = {
        "reference_notebook": {
            "path": str(REFERENCE_NOTEBOOK),
            "expected_sha256": EXPECTED_REFERENCE_NOTEBOOK_SHA256,
            "actual_sha256": sha256_file(REFERENCE_NOTEBOOK),
        },
        "stage1_notebook": {
            "path": str(STAGE1_NOTEBOOK),
            "expected_sha256": EXPECTED_STAGE1_SHA256,
            "actual_sha256": sha256_file(STAGE1_NOTEBOOK),
        },
    }
    for item in audit.values():
        item["match"] = item["expected_sha256"] == item["actual_sha256"]
    if not all(item["match"] for item in audit.values()):
        raise RuntimeError(f"Frozen reference integrity failure: {audit}")
    return audit


def read_split(path):
    frame = pd.read_csv(path, dtype={"ImageID": str, "Subject": str})
    required = {"ImageID", "Subject", "Group", "GroupLabel", "BiasCorrPath"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path} missing columns: {sorted(missing_columns)}")
    if frame[list(required)].isna().any().any() or frame.ImageID.duplicated().any():
        raise ValueError(f"Invalid persisted split: {path}")
    mapped = frame.Group.map(LABEL_MAP)
    if mapped.isna().any() or not np.array_equal(mapped.astype(int), frame.GroupLabel.astype(int)):
        raise ValueError(f"Label mapping mismatch: {path}")
    missing_files = [value for value in frame.BiasCorrPath if not Path(value).is_file()]
    if missing_files:
        raise FileNotFoundError(missing_files[0])
    return frame


class Fold1MRIDataset(Dataset):
    """Exact reference preprocessing; intentionally no augmentation."""

    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = np.asarray(nib.load(row.BiasCorrPath).dataobj, dtype=np.float32)
        if image.shape != (128, 128, 128):
            raise ValueError(f"Unexpected MRI shape {image.shape}: {row.BiasCorrPath}")
        image = np.nan_to_num(image)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        if not np.isfinite(image).all():
            raise FloatingPointError(f"Non-finite preprocessed MRI: {row.BiasCorrPath}")
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0),
            "label": torch.tensor(int(row.GroupLabel), dtype=torch.long),
            "ImageID": str(row.ImageID),
        }


def seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_model(key):
    if PRETRAINED:
        raise RuntimeError("Pretrained weights are forbidden for this scratch comparison")
    if key == "D":
        return DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=NUM_CLASSES,
            pretrained=False,
        )
    if key == "E":
        return EfficientNetBN(
            "efficientnet-b0",
            pretrained=False,
            spatial_dims=3,
            in_channels=1,
            num_classes=NUM_CLASSES,
        )
    raise ValueError(key)


def state_checksum(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def autocast_context():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def calculate_metrics(y_true, probabilities):
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=np.arange(NUM_CLASSES),
        zero_division=0,
    )
    try:
        macro_auc = float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=np.arange(NUM_CLASSES),
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        macro_auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "macro_auc": macro_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "predictions": predictions,
        "predicted_counts": np.bincount(predictions, minlength=NUM_CLASSES),
        "probabilities": probabilities,
    }


def run_phase(model, loader, criterion, training, optimizer, device, epoch):
    model.train(training)
    total_loss = 0.0
    labels_all, probabilities_all, imageids_all = [], [], []
    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if not torch.isfinite(images).all():
            raise FloatingPointError(f"Non-finite input: epoch={epoch}, batch={batch_index}")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            logits = model(images)
            loss = criterion(logits, labels)
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite logits/loss: epoch={epoch}, batch={batch_index}")
        if training:
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite gradients: epoch={epoch}, batch={batch_index}")
            optimizer.step()
        probabilities = torch.softmax(logits.float(), dim=1)
        total_loss += float(loss.item()) * len(labels)
        labels_all.append(labels.detach().cpu().numpy())
        probabilities_all.append(probabilities.detach().cpu().numpy())
        imageids_all.extend(map(str, batch["ImageID"]))
    y_true = np.concatenate(labels_all)
    probability_array = np.concatenate(probabilities_all)
    metrics = calculate_metrics(y_true, probability_array)
    metrics["loss"] = total_loss / len(loader.dataset)
    metrics["imageids"] = imageids_all
    return metrics


def save_plots(results_dir, history, y_true, predictions, probabilities, model_name):
    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(results_dir / filename, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)

    matrix = confusion_matrix(y_true, predictions, labels=np.arange(NUM_CLASSES))
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(results_dir / "confusion_matrix.csv")
    pd.DataFrame(normalized, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(results_dir / "normalized_confusion_matrix.csv")
    for values, normalized_flag, filename in [
        (matrix, False, "confusion_matrix.png"),
        (normalized, True, "normalized_confusion_matrix.png"),
    ]:
        fig, axis = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            values,
            annot=True,
            fmt=".2f" if normalized_flag else "d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=axis,
        )
        axis.set(
            xlabel="Predicted label",
            ylabel="True label",
            title=f"Fold-1 validation {'normalized ' if normalized_flag else ''}confusion matrix — {model_name}",
        )
        save(fig, filename)

    roc_rows = []
    fig, axis = plt.subplots(figsize=(7, 6))
    for class_index, class_name in enumerate(CLASS_NAMES):
        binary_truth = (np.asarray(y_true) == class_index).astype(int)
        fpr, tpr, thresholds = roc_curve(binary_truth, probabilities[:, class_index])
        auc_value = roc_auc_score(binary_truth, probabilities[:, class_index])
        axis.plot(fpr, tpr, label=f"{class_name} (AUC={auc_value:.3f})")
        roc_rows.extend(
            {"class": class_name, "fpr": float(x), "tpr": float(y), "threshold": float(z)}
            for x, y, z in zip(fpr, tpr, thresholds)
        )
    pd.DataFrame(roc_rows).to_csv(results_dir / "roc_data.csv", index=False)
    axis.plot([0, 1], [0, 1], "k--", alpha=0.5)
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title=f"Fold-1 validation ROC — {model_name}")
    axis.legend(loc="lower right", fontsize=8)
    axis.grid(alpha=0.25)
    save(fig, "roc_curves.png")

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.ravel()
    for column, label in [("train_loss", "Train"), ("val_loss", "Validation")]:
        axes[0].plot(history.epoch, history[column], label=label)
    for column, label in [("train_macro_f1", "Train"), ("val_macro_f1", "Validation")]:
        axes[1].plot(history.epoch, history[column], label=label)
    axes[2].plot(history.epoch, history.val_macro_auc)
    axes[3].plot(history.epoch, history.val_mcc)
    axes[4].plot(history.epoch, history.learning_rate)
    axes[5].plot(history.epoch, history.train_val_macro_f1_gap)
    labels = [
        ("Loss", "Cross entropy"),
        ("Macro F1", "Macro F1"),
        ("Validation Macro AUC", "Macro AUC"),
        ("Validation MCC", "MCC"),
        ("Learning rate", "LR"),
        ("Generalization gap", "Train − validation Macro F1"),
    ]
    for axis, (title, ylabel) in zip(axes, labels):
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        axis.grid(alpha=0.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
    fig.suptitle(f"Fold-1 learning curves — {model_name}")
    save(fig, "learning_curves.png")


def run_baseline(key):
    spec = ARCHITECTURES[key]
    results_dir = spec["results"]
    results_dir.mkdir(parents=True, exist_ok=True)
    if any((results_dir / name).exists() for name in ["best_model.pt", "history.csv", "validation_metrics.csv"]):
        raise RuntimeError(f"Fresh run required; artifacts already exist in {results_dir}")
    if PRETRAINED or TEST_INFERENCE_PERFORMED or BATCH_SIZE != 4 or EFFECTIVE_BATCH_SIZE != 4:
        raise RuntimeError("Frozen scratch architecture-baseline configuration was altered")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly physical GPU 2 must be exposed")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 support is required")
    device = torch.device("cuda:0")
    frozen_reference_audit = audit_frozen_references()

    train_csv, val_csv, test_csv = (FOLD_DIR / name for name in ("train.csv", "val.csv", "test.csv"))
    if sha256_file(train_csv) != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("Fold-1 training assignment changed")
    if sha256_file(val_csv) != EXPECTED_VAL_SHA256:
        raise RuntimeError("Fold-1 validation assignment changed")
    if sha256_file(test_csv) != EXPECTED_TEST_SHA256:
        raise RuntimeError("Fixed test membership assignment changed")
    train_df = read_split(train_csv)
    val_df = read_split(val_csv)
    # Only membership IDs are opened before validation screening ends.
    test_ids = pd.read_csv(test_csv, usecols=["ImageID"], dtype={"ImageID": str}).ImageID
    train_ids, val_ids, test_id_set = set(train_df.ImageID), set(val_df.ImageID), set(test_ids)
    if len(train_df) != 2298 or len(val_df) != 575 or len(test_ids) != 719:
        raise RuntimeError("Unexpected persisted Fold-1 sizes")
    overlaps = {
        "train_validation": len(train_ids & val_ids),
        "train_test": len(train_ids & test_id_set),
        "validation_test": len(val_ids & test_id_set),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"ImageID leakage detected: {overlaps}")
    if train_df.ImageID.nunique() != 2298 or val_df.ImageID.nunique() != 575 or test_ids.nunique() != 719:
        raise RuntimeError("ImageID uniqueness audit failed")

    split_rows = []
    for split_name, frame, checksum in [
        ("train", train_df, EXPECTED_TRAIN_SHA256),
        ("validation", val_df, EXPECTED_VAL_SHA256),
    ]:
        counts = frame.Group.value_counts().reindex(CLASS_NAMES, fill_value=0)
        split_rows.append({
            "split": split_name,
            "rows": len(frame),
            "unique_imageids": frame.ImageID.nunique(),
            "csv_sha256": checksum,
            "exact_validated_resnet_fold1_assignment": True,
            **{f"count_{name}": int(counts[name]) for name in CLASS_NAMES},
        })
    pd.DataFrame(split_rows).to_csv(results_dir / "fold1_split_audit.csv", index=False)

    example = np.asarray(nib.load(train_df.iloc[0].BiasCorrPath).dataobj, dtype=np.float32)
    processed = np.nan_to_num(example)
    processed = (processed - processed.min()) / (processed.max() - processed.min() + 1e-8)
    if example.shape != (128, 128, 128) or not np.isfinite(processed).all():
        raise RuntimeError("Reference preprocessing audit failed")
    json_dump(results_dir / "data_equivalence.json", {
        "dataset": "same 3,592-ImageID universe as validated scratch ResNet baseline",
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "fixed_test_ids": len(test_ids),
        "overlap_counts": overlaps,
        "train_csv_sha256": EXPECTED_TRAIN_SHA256,
        "validation_csv_sha256": EXPECTED_VAL_SHA256,
        "test_csv_sha256": EXPECTED_TEST_SHA256,
        "test_membership_ids_opened_for_leakage_audit_only": True,
        "test_labels_opened": False,
        "test_paths_opened": False,
        "test_volumes_opened": False,
        "test_metrics_calculated": False,
        "normalization": "np.nan_to_num then per-volume min-max",
        "tensor_shape": [1, 128, 128, 128],
    })

    print(f"{spec['model_name']} Fold-1 scratch architecture baseline")
    print(f"MONAI={monai.__version__}; PyTorch={torch.__version__}")
    print(f"GPU={torch.cuda.get_device_name(0)}; physical UUID={GPU2_UUID}; BF16=True")
    print(f"Label mapping={LABEL_MAP}")
    print(f"Fold-1 exact persisted assignment: train={len(train_df)}, validation={len(val_df)}")
    print("Fixed test: ImageID membership leakage audit only; evaluation=False")

    train_ds, val_ds = Fold1MRIDataset(train_df), Fold1MRIDataset(val_df)
    loader_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=loader_generator,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        worker_init_fn=seed_worker,
    )
    if not isinstance(train_loader.sampler, RandomSampler):
        raise RuntimeError("Natural shuffled RandomSampler is required")

    seed_all(SEED)
    model = build_model(key)
    initialization_sha256 = state_checksum(model)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if total_parameters != trainable_parameters:
        raise RuntimeError("All scratch parameters must be trainable")
    model = model.to(device)

    # Mandatory official-architecture dummy audit with [1,1,128,128,128].
    model.eval()
    dummy = torch.zeros(1, 1, 128, 128, 128, device=device)
    with torch.no_grad(), autocast_context():
        dummy_output = model(dummy)
    if tuple(dummy_output.shape) != (1, NUM_CLASSES) or not torch.isfinite(dummy_output).all():
        raise RuntimeError(f"Dummy output audit failed: {tuple(dummy_output.shape)}")
    del dummy, dummy_output

    # BF16 numerical stability check includes a real backward pass.
    model.train()
    preflight_images = torch.stack([val_ds[0]["image"], val_ds[1]["image"]]).to(device)
    preflight_labels = torch.tensor(
        [int(val_df.iloc[0].GroupLabel), int(val_df.iloc[1].GroupLabel)],
        device=device,
    )
    preflight_criterion = nn.CrossEntropyLoss()
    model.zero_grad(set_to_none=True)
    with autocast_context():
        preflight_logits = model(preflight_images)
        preflight_loss = preflight_criterion(preflight_logits, preflight_labels)
    preflight_loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not torch.isfinite(preflight_loss) or not torch.isfinite(preflight_logits).all():
        raise FloatingPointError("BF16 preflight produced non-finite logits/loss")
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("BF16 preflight produced non-finite gradients")
    del model, preflight_images, preflight_labels, preflight_logits, preflight_loss, gradients, preflight_criterion
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{spec['model_name'].upper()} BASELINE INTEGRITY")
    checks = {
        "official MONAI architecture": True,
        "scratch initialization": not PRETRAINED,
        "six outputs": True,
        "plain CrossEntropyLoss": True,
        "natural shuffle with no sampler": isinstance(train_loader.sampler, RandomSampler),
        "no proposed pipeline": True,
        "no augmentation": True,
        "same Fold-1 split": True,
        "same preprocessing": True,
        "BF16 forward/backward": True,
    }
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    if not all(checks.values()):
        raise RuntimeError(f"Model integrity failed: {checks}")
    print(f"Architecture={spec['architecture']}")
    print(f"Input shape=[1,1,128,128,128]; output shape=[1,6]")
    print(f"Total/trainable parameters={total_parameters:,}/{trainable_parameters:,}")

    # Reconstruct the exact fresh seed-42 model after preflight.
    seed_all(SEED)
    model = build_model(key)
    if state_checksum(model) != initialization_sha256:
        raise RuntimeError("Fresh scratch initialization is not deterministic")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    if criterion.weight is not None or float(criterion.label_smoothing) != 0.0:
        raise RuntimeError("Loss contamination detected")
    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=1 / 3,
        patience=3,
    )

    history_rows, per_class_history_rows, health_events = [], [], []
    best_macro_f1, best_epoch, patience_counter = -np.inf, None, 0
    training_started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    print("\nTRAINING START")
    print(
        f"Natural shuffle; CrossEntropyLoss(no weights/no smoothing); no augmentation; "
        f"AdamW lr={INITIAL_LR:.1e}; wd={WEIGHT_DECAY:.1e}; batch={BATCH_SIZE}; "
        f"max_epochs={MAX_EPOCHS}; patience={EARLY_STOP_PATIENCE}; BF16=True"
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_started = time.time()
        train_metrics = run_phase(model, train_loader, criterion, True, optimizer, device, epoch)
        with torch.no_grad():
            val_metrics = run_phase(model, val_loader, criterion, False, None, device, epoch)
        epoch_minutes = (time.time() - epoch_started) / 60.0
        predicted_counts = val_metrics["predicted_counts"]
        predicted_classes = int(np.count_nonzero(predicted_counts))
        dominant_fraction = float(predicted_counts.max() / predicted_counts.sum())
        events = []
        if dominant_fraction > 0.90:
            events.append("prediction_concentration_above_90_percent")
        if epoch >= 10 and predicted_classes < NUM_CLASSES:
            events.append("fewer_than_six_predicted_classes_after_epoch_10")
        for event in events:
            health_events.append({"epoch": epoch, "event": event, "technically_fatal": False})
            print(f"HEALTH WARNING epoch={epoch}: {event}")

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_macro_precision": train_metrics["macro_precision"],
            "val_macro_precision": val_metrics["macro_precision"],
            "train_macro_recall": train_metrics["macro_recall"],
            "val_macro_recall": val_metrics["macro_recall"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_mcc": val_metrics["mcc"],
            "val_macro_auc": val_metrics["macro_auc"],
            "train_val_macro_f1_gap": train_metrics["macro_f1"] - val_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation_predicted_classes": predicted_classes,
            "dominant_prediction_fraction": dominant_fraction,
            "epoch_minutes": epoch_minutes,
        }
        for class_index, class_name in enumerate(CLASS_NAMES):
            class_row = {
                "epoch": epoch,
                "class": class_name,
                "precision": float(val_metrics["precision"][class_index]),
                "recall": float(val_metrics["recall"][class_index]),
                "f1": float(val_metrics["f1"][class_index]),
                "support": int(val_metrics["support"][class_index]),
                "predicted_count": int(predicted_counts[class_index]),
            }
            per_class_history_rows.append(class_row)
            for metric_name in ("precision", "recall", "f1", "support", "predicted_count"):
                row[f"val_{class_name}_{metric_name}"] = class_row[metric_name]
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(results_dir / "history.csv", index=False)
        pd.DataFrame(per_class_history_rows).to_csv(results_dir / "per_class_history.csv", index=False)

        improved = np.isfinite(val_metrics["macro_f1"]) and val_metrics["macro_f1"] > best_macro_f1 + MIN_DELTA
        if improved:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_macro_f1": best_macro_f1,
                "checkpoint_selection": "validation_macro_f1_only",
                "architecture_key": key,
                "architecture": spec["architecture"],
                "pretrained": False,
                "label_mapping": LABEL_MAP,
                "train_csv_sha256": EXPECTED_TRAIN_SHA256,
                "validation_csv_sha256": EXPECTED_VAL_SHA256,
                "test_inference_performed": False,
            }, results_dir / "best_model.pt")
        else:
            patience_counter += 1
        scheduler.step(val_metrics["macro_f1"])
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_macro_f1,
            "resume_default": False,
        }, results_dir / "latest_checkpoint.pt")

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"loss={train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
            f"acc={train_metrics['accuracy']:.4f}/{val_metrics['accuracy']:.4f} | "
            f"P={train_metrics['macro_precision']:.4f}/{val_metrics['macro_precision']:.4f} | "
            f"R={train_metrics['macro_recall']:.4f}/{val_metrics['macro_recall']:.4f} | "
            f"F1={train_metrics['macro_f1']:.4f}/{val_metrics['macro_f1']:.4f} | "
            f"MCC={val_metrics['mcc']:.4f} AUC={val_metrics['macro_auc']:.4f} | "
            f"classes={predicted_classes}/6 counts={predicted_counts.tolist()} | "
            f"gap={row['train_val_macro_f1_gap']:.4f} lr={row['learning_rate']:.2e} time={epoch_minutes:.2f}m"
        )
        print("  class F1: " + " ".join(
            f"{class_name}={val_metrics['f1'][class_index]:.4f}"
            for class_index, class_name in enumerate(CLASS_NAMES)
        ))
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(
                f"Early stopping at epoch {epoch}; best epoch={best_epoch}; "
                f"best validation Macro F1={best_macro_f1:.6f}"
            )
            break

    training_minutes = (time.time() - training_started) / 60.0
    if best_epoch is None or not (results_dir / "best_model.pt").is_file():
        raise RuntimeError("No valid validation-Macro-F1 checkpoint was produced")
    history = pd.DataFrame(history_rows)

    best_checkpoint = torch.load(results_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    torch.cuda.synchronize()
    inference_started = time.time()
    with torch.no_grad():
        best_val_metrics = run_phase(model, val_loader, criterion, False, None, device, best_epoch)
    torch.cuda.synchronize()
    inference_seconds = time.time() - inference_started
    if abs(float(best_val_metrics["macro_f1"]) - float(best_checkpoint["best_val_macro_f1"])) > 1e-10:
        raise RuntimeError("Reloaded best-checkpoint Macro F1 differs from checkpoint metadata")
    if set(best_val_metrics["imageids"]) != set(val_df.ImageID.astype(str)):
        raise RuntimeError("Reloaded checkpoint validation ImageIDs differ from persisted Fold 1")
    if np.count_nonzero(best_val_metrics["predicted_counts"]) != NUM_CLASSES:
        raise RuntimeError("Authoritative checkpoint is technically collapsed: not all six classes predicted")

    position = {imageid: index for index, imageid in enumerate(best_val_metrics["imageids"])}
    order = np.array([position[str(imageid)] for imageid in val_df.ImageID])
    predictions = best_val_metrics["predictions"][order]
    probabilities = best_val_metrics["probabilities"][order]
    y_true = val_df.GroupLabel.astype(int).to_numpy()
    prediction_frame = val_df[["ImageID", "Subject", "Group", "GroupLabel", "BiasCorrPath"]].copy()
    prediction_frame["predicted_index"] = predictions
    prediction_frame["predicted_label"] = [CLASS_NAMES[index] for index in predictions]
    for class_index, class_name in enumerate(CLASS_NAMES):
        prediction_frame[f"prob_{class_name}"] = probabilities[:, class_index]
    prediction_frame.to_csv(results_dir / "validation_predictions.csv", index=False)

    report = classification_report(
        y_true,
        predictions,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.reset_index().rename(columns={"index": "class_or_average"}).to_csv(
        results_dir / "classification_report.csv", index=False
    )
    per_class = pd.DataFrame({
        "class": CLASS_NAMES,
        "precision": best_val_metrics["precision"],
        "recall": best_val_metrics["recall"],
        "f1": best_val_metrics["f1"],
        "predicted": best_val_metrics["predicted_counts"],
        "support": best_val_metrics["support"],
    })
    per_class.to_csv(results_dir / "per_class_metrics.csv", index=False)
    best_history_row = history.loc[history.epoch == best_epoch].iloc[0]
    validation_metrics = {
        "best_epoch": int(best_epoch),
        "accuracy": best_val_metrics["accuracy"],
        "macro_precision": best_val_metrics["macro_precision"],
        "macro_recall": best_val_metrics["macro_recall"],
        "macro_f1": best_val_metrics["macro_f1"],
        "mcc": best_val_metrics["mcc"],
        "macro_auc": best_val_metrics["macro_auc"],
        "train_macro_f1_at_best_epoch": float(best_history_row.train_macro_f1),
        "train_validation_macro_f1_gap": float(best_history_row.train_val_macro_f1_gap),
        "predicted_classes": int(np.count_nonzero(best_val_metrics["predicted_counts"])),
        "test_inference_performed": False,
    }
    pd.DataFrame([validation_metrics]).to_csv(results_dir / "validation_metrics.csv", index=False)
    save_plots(results_dir, history, y_true, predictions, probabilities, spec["model_name"])

    checkpoint_size_mib = (results_dir / "best_model.pt").stat().st_size / 2**20
    peak_gpu_memory_gib = torch.cuda.max_memory_allocated(device) / 2**30
    runtime = {
        "valid": True,
        "precision_mode": "bf16",
        "gpu": torch.cuda.get_device_name(0),
        "physical_gpu_uuid": GPU2_UUID,
        "epochs_completed": int(history.epoch.max()),
        "best_epoch": int(best_epoch),
        "average_epoch_minutes": float(history.epoch_minutes.mean()),
        "total_training_minutes": float(training_minutes),
        "peak_gpu_memory_gib": float(peak_gpu_memory_gib),
        "validation_inference_seconds": float(inference_seconds),
        "inference_seconds_per_mri": float(inference_seconds / len(val_df)),
        "batch_size": BATCH_SIZE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "num_workers": NUM_WORKERS,
    }
    json_dump(results_dir / "runtime_summary.json", runtime)
    pd.DataFrame([{
        "model": spec["model_name"],
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "checkpoint_size_mib": checkpoint_size_mib,
        "peak_gpu_memory_gib": peak_gpu_memory_gib,
        "average_epoch_minutes": runtime["average_epoch_minutes"],
        "total_training_minutes": training_minutes,
        "inference_seconds_per_mri": runtime["inference_seconds_per_mri"],
    }]).to_csv(results_dir / "model_complexity.csv", index=False)
    json_dump(results_dir / "health_checks.json", {
        "valid": True,
        "finite_inputs_logits_losses_gradients": True,
        "authoritative_checkpoint_predicts_all_six_classes": True,
        "authoritative_checkpoint_dominant_prediction_fraction": float(best_val_metrics["predicted_counts"].max() / len(val_df)),
        "events": health_events,
        "technical_collapse": False,
    })
    json_dump(results_dir / "config.json", {
        "experiment": f"{spec['model_name']} ImageID Fold-1 official architecture baseline",
        "architecture": spec["architecture"],
        "monai_version": monai.__version__,
        "pretrained": False,
        "seed": SEED,
        "label_mapping": LABEL_MAP,
        "train_csv_sha256": EXPECTED_TRAIN_SHA256,
        "validation_csv_sha256": EXPECTED_VAL_SHA256,
        "normalization": "np.nan_to_num then per-volume min-max",
        "augmentation": "none",
        "sampler": "natural shuffled RandomSampler; no weighting",
        "loss": "plain CrossEntropyLoss; no class weights/focal/label smoothing",
        "optimizer": "AdamW",
        "initial_lr": INITIAL_LR,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "ReduceLROnPlateau(mode=max, validation Macro F1, factor=1/3, patience=3)",
        "batch_size": BATCH_SIZE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "checkpoint_selection": "validation Macro F1 only",
        "precision": "bf16",
        "initialization_sha256": initialization_sha256,
        "test_membership_ids_read_for_leakage_audit_only": True,
        "test_labels_paths_or_volumes_opened": False,
        "test_inference_performed": False,
        "frozen_reference_audit": frozen_reference_audit,
    })
    json_dump(results_dir / "integrity_complete.json", {
        "valid": True,
        "official_monai_architecture": True,
        "scratch_initialization": True,
        "plain_cross_entropy": True,
        "natural_sampling": True,
        "no_augmentation": True,
        "no_proposed_components": True,
        "same_fold1_imageids": True,
        "same_preprocessing": True,
        "checkpoint_reloaded": True,
        "checkpoint_macro_f1_reproduced": True,
        "all_six_classes_predicted": True,
        "test_inference_performed": False,
    })

    print("\nAUTHORITATIVE RELOADED FOLD-1 VALIDATION RESULT")
    print(pd.DataFrame([validation_metrics]).to_string(index=False))
    print("\nCLASS-WISE RESULT")
    print(per_class.to_string(index=False))
    print("\nRUNTIME/COMPLEXITY")
    print(json.dumps(runtime, indent=2))
    print("RUN_VALID; fixed test evaluated=False")

    del best_checkpoint, model, optimizer, scheduler, criterion
    del train_loader, val_loader, train_ds, val_ds
    del train_df, val_df, prediction_frame, predictions, probabilities, y_true, best_val_metrics
    plt.close("all")
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    print(f"{spec['model_name']} Python cleanup complete; CUDA cache cleared")


def gpu_snapshot():
    output = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ], text=True)
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 6 and parts[1] == GPU2_UUID:
            return {
                "index": int(parts[0]),
                "uuid": parts[1],
                "memory_used_mib": int(float(parts[2])),
                "utilization_percent": int(float(parts[3])),
                "power_draw_w": float(parts[4]),
                "power_limit_w": float(parts[5]),
                "timestamp_utc": utc_now(),
            }
    raise RuntimeError(f"Physical GPU 2 UUID not found: {GPU2_UUID}")


def gpu_compute_processes():
    completed = subprocess.run([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True, capture_output=True, check=True)
    processes = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4 and parts[0] == GPU2_UUID:
            processes.append({
                "gpu_uuid": parts[0],
                "pid": int(parts[1]),
                "process_name": parts[2],
                "used_memory_mib": int(float(parts[3])),
            })
    return processes


def wait_for_gpu2_idle(stage):
    deadline = time.time() + GPU_IDLE_TIMEOUT_SECONDS
    samples = []
    while True:
        snapshot = gpu_snapshot()
        processes = gpu_compute_processes()
        snapshot["compute_processes"] = processes
        samples.append(snapshot)
        if snapshot["memory_used_mib"] <= GPU_IDLE_THRESHOLD_MIB and not processes:
            print(
                f"GPU 2 idle at {stage}: memory={snapshot['memory_used_mib']} MiB, "
                f"utilization={snapshot['utilization_percent']}%, processes=0"
            )
            return snapshot, samples
        if time.time() >= deadline:
            raise RuntimeError(f"GPU 2 did not become idle at {stage}: {snapshot}")
        print(f"Waiting for GPU 2 at {stage}: memory={snapshot['memory_used_mib']} MiB, processes={len(processes)}")
        time.sleep(5)


def execute_notebook(key):
    spec = ARCHITECTURES[key]
    jupyter = Path(sys.executable).parent / "jupyter"
    if not jupyter.is_file():
        raise FileNotFoundError(jupyter)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = GPU2_UUID
    environment["PYTHONUNBUFFERED"] = "1"
    command = [
        str(jupyter),
        "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=-1",
        "--ExecutePreprocessor.kernel_name=python3",
        str(spec["notebook"]),
    ]
    print(f"Launching {spec['model_name']}: {' '.join(command)}")
    started = utc_now()
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment)
    finished = utc_now()
    if completed.returncode != 0:
        raise RuntimeError(f"{spec['model_name']} failed with exit code {completed.returncode}")
    return {"returncode": completed.returncode, "started_utc": started, "finished_utc": finished}


def notebook_epoch_lines(path):
    import nbformat

    notebook = nbformat.read(path, as_version=4)
    errors = [
        output
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if errors:
        raise RuntimeError(f"Executed notebook contains errors: {path}")
    return sum(
        1
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
        for line in output.get("text", "").splitlines()
        if line.startswith("Epoch ")
    )


def validate_outputs(key):
    spec = ARCHITECTURES[key]
    missing = [name for name in REQUIRED_OUTPUTS if not (spec["results"] / name).is_file()]
    if missing:
        raise RuntimeError(f"{spec['model_name']} missing outputs: {missing}")
    metrics = pd.read_csv(spec["results"] / "validation_metrics.csv")
    predictions = pd.read_csv(spec["results"] / "validation_predictions.csv", dtype={"ImageID": str})
    integrity = json.loads((spec["results"] / "integrity_complete.json").read_text(encoding="utf-8"))
    if len(metrics) != 1 or len(predictions) != 575 or predictions.ImageID.nunique() != 575:
        raise RuntimeError(f"{spec['model_name']} output row audit failed")
    if bool(metrics.test_inference_performed.iloc[0]) or not integrity.get("valid"):
        raise RuntimeError(f"{spec['model_name']} integrity/test-seal audit failed")
    return metrics.iloc[0].to_dict()


def create_comparison_and_decisions():
    direct = pd.read_csv(REFERENCE_RESULTS / "direct_comparison.csv").set_index("metric")
    resnet_metrics = {
        metric: float(direct.loc[metric, "scratch_3d_resnet_fold1"])
        for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "mcc", "macro_auc")
    }
    resnet_class = pd.read_csv(REFERENCE_RESULTS / "per_class_metrics.csv")
    resnet_runtime = json.loads((REFERENCE_RESULTS / "runtime_summary.json").read_text(encoding="utf-8"))

    rows = [{
        "Model": "Vanilla scratch 3D ResNet-18",
        "Accuracy": resnet_metrics["accuracy"],
        "Macro Precision": resnet_metrics["macro_precision"],
        "Macro Recall": resnet_metrics["macro_recall"],
        "Macro F1": resnet_metrics["macro_f1"],
        "MCC": resnet_metrics["mcc"],
        "Macro AUC": resnet_metrics["macro_auc"],
        **{f"{name} F1": float(resnet_class.loc[resnet_class["class"] == name, "f1"].iloc[0]) for name in CLASS_NAMES},
        "Params": int(resnet_runtime["total_parameters"]),
        "Runtime": float(resnet_runtime["total_training_minutes"]),
        "Peak GPU Memory": float(resnet_runtime["peak_gpu_memory_gib"]),
    }]
    decisions = {}
    for key in EXECUTION_ORDER:
        spec = ARCHITECTURES[key]
        metrics = pd.read_csv(spec["results"] / "validation_metrics.csv").iloc[0]
        per_class = pd.read_csv(spec["results"] / "per_class_metrics.csv")
        complexity = pd.read_csv(spec["results"] / "model_complexity.csv").iloc[0]
        rows.append({
            "Model": spec["model_name"],
            "Accuracy": float(metrics.accuracy),
            "Macro Precision": float(metrics.macro_precision),
            "Macro Recall": float(metrics.macro_recall),
            "Macro F1": float(metrics.macro_f1),
            "MCC": float(metrics.mcc),
            "Macro AUC": float(metrics.macro_auc),
            **{f"{name} F1": float(per_class.loc[per_class["class"] == name, "f1"].iloc[0]) for name in CLASS_NAMES},
            "Params": int(complexity.total_parameters),
            "Runtime": float(complexity.total_training_minutes),
            "Peak GPU Memory": float(complexity.peak_gpu_memory_gib),
        })
        difference = float(metrics.macro_f1 - resnet_metrics["macro_f1"])
        prefix = spec["decision_prefix"]
        if difference > 0:
            decision = f"{prefix}_THREATENS_RESNET"
        elif abs(difference) <= 0.005:
            decision = f"{prefix}_COMPARABLE_TO_RESNET"
        else:
            decision = f"{prefix}_BELOW_RESNET"
        decisions[key] = {
            "decision": decision,
            "macro_f1": float(metrics.macro_f1),
            "resnet_macro_f1": resnet_metrics["macro_f1"],
            "macro_f1_difference": difference,
            "screening_only": True,
            "additional_folds_launched": False,
        }
        json_dump(spec["results"] / "screening_decision.json", decisions[key])
    comparison = pd.DataFrame(rows)
    comparison.to_csv(BASELINE_RESULTS_ROOT / "Fold1_Established_Architecture_Comparison.csv", index=False)
    json_dump(BASELINE_RESULTS_ROOT / "Fold1_Established_Architecture_Screening_Decisions.json", decisions)
    return comparison, decisions


def main():
    BASELINE_NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    initial_root_hashes = snapshot_preexisting_root_files()
    sequence = {
        "execution_order": EXECUTION_ORDER,
        "strictly_sequential": True,
        "physical_gpu_uuid": GPU2_UUID,
        "started_utc": utc_now(),
        "test_membership_ids_only_before_and_during_training": True,
        "test_labels_paths_volumes_predictions_metrics_accessed": False,
        "runs": {},
        "preexisting_root_file_hashes_before": initial_root_hashes,
    }
    audit_path = BASELINE_RESULTS_ROOT / "Fold1_DenseNet_EfficientNet_Sequential_Audit.json"
    try:
        audit_frozen_references()
        for key in EXECUTION_ORDER:
            if not ARCHITECTURES[key]["notebook"].is_file():
                raise FileNotFoundError(ARCHITECTURES[key]["notebook"])
        initial_gpu, initial_samples = wait_for_gpu2_idle("before DenseNet")
        sequence["initial_gpu"] = initial_gpu
        sequence["initial_gpu_samples"] = initial_samples
        for key in EXECUTION_ORDER:
            spec = ARCHITECTURES[key]
            if any((spec["results"] / name).exists() for name in ["best_model.pt", "history.csv", "validation_metrics.csv"]):
                raise RuntimeError(f"Fresh result directory required: {spec['results']}")
            entry = execute_notebook(key)
            entry["metrics"] = validate_outputs(key)
            entry["epoch_output_lines"] = notebook_epoch_lines(spec["notebook"])
            released, samples = wait_for_gpu2_idle(f"after {spec['model_name']} process exit")
            entry["process_exited"] = True
            entry["released_gpu"] = released
            entry["gpu_release_samples"] = samples
            sequence["runs"][key] = entry
            json_dump(audit_path, sequence)

        comparison, decisions = create_comparison_and_decisions()
        final_root_hashes = snapshot_preexisting_root_files()
        if final_root_hashes != initial_root_hashes:
            raise RuntimeError("A pre-existing root notebook/script changed during new baseline execution")
        final_gpu, final_samples = wait_for_gpu2_idle("after all Fold-1 architecture baselines")
        sequence.update({
            "completed_utc": utc_now(),
            "valid": True,
            "comparison_path": str(BASELINE_RESULTS_ROOT / "Fold1_Established_Architecture_Comparison.csv"),
            "decisions": decisions,
            "preexisting_root_file_hashes_after": final_root_hashes,
            "preexisting_files_unchanged": True,
            "final_gpu": final_gpu,
            "final_gpu_samples": final_samples,
            "additional_folds_launched": False,
        })
        json_dump(audit_path, sequence)
        print("\nFOLD-1 ESTABLISHED ARCHITECTURE COMPARISON")
        print(comparison.to_string(index=False))
        print("\nSCREENING DECISIONS")
        print(json.dumps(decisions, indent=2))
    except Exception as exc:
        sequence.update({
            "completed_utc": utc_now(),
            "valid": False,
            "failure": f"{type(exc).__name__}: {exc}",
            "efficientnet_started_after_densenet_failure": False,
        })
        json_dump(audit_path, sequence)
        print(f"ARCHITECTURE_BASELINE_PIPELINE_INVALID: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
