"""
Face Emotion Recognition — Training Script (PyTorch + CUDA)
===========================================================
- PyTorch: full CUDA support trên Windows native
- Export ONNX sau khi train → dùng với onnxruntime (nhất quán với face detector)
- Log đầy đủ metrics ra console + CSV

Usage:
    python train.py --data D:/Face_Emotion_Recognition/archive --epochs 50 --batch 256
    python train.py --data D:/Face_Emotion_Recognition/archive --epochs 50 --batch 256 --no-tune
"""

import os
import sys
import json
import argparse
import time
import csv
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
EMOTIONS   = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE   = 48
OUTPUT_DIR = 'models'
RESULTS_DIR = 'results'
LOG_FILE   = 'results/training_log.csv'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',    type=str, required=True)
    p.add_argument('--epochs',  type=int, default=50)
    p.add_argument('--batch',   type=int, default=256)
    p.add_argument('--lr',      type=float, default=1e-3)
    p.add_argument('--dropout', type=float, default=0.4)
    p.add_argument('--filters', type=int, default=64)
    p.add_argument('--dense',   type=int, default=512)
    p.add_argument('--no-tune', action='store_true')
    p.add_argument('--tune-trials', type=int, default=10)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint để train thêm (vd: models/best.pt)')
    return p.parse_args()


# ─────────────────────────────────────────────
#  GPU CHECK
# ─────────────────────────────────────────────
def check_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu = torch.cuda.get_device_properties(0)
        print(f"\n{'='*55}")
        print(f"  GPU : {gpu.name}")
        print(f"  VRAM: {gpu.total_memory / 1024**3:.1f} GB")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"{'='*55}")
    else:
        device = torch.device('cpu')
        print("\n[WARNING] No GPU found — training on CPU (slow!)")
    return device


# ─────────────────────────────────────────────
#  DATA LOADERS
# ─────────────────────────────────────────────
def build_loaders(train_dir, test_dir, batch_size):
    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    val_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    full_train = datasets.ImageFolder(train_dir, transform=train_transform)
    n_total    = len(full_train)
    n_val      = int(n_total * 0.1)
    n_train    = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Gán val_transform cho val_ds
    val_ds.dataset = datasets.ImageFolder(train_dir, transform=val_transform)

    test_ds = datasets.ImageFolder(test_dir, transform=val_transform)

    # WeightedRandomSampler — xử lý class imbalance khi sampling
    targets = [full_train.targets[i] for i in train_ds.indices]
    classes = np.array(targets)
    class_counts = np.bincount(classes)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[classes]
    sampler = WeightedRandomSampler(
        weights=torch.FloatTensor(sample_weights),
        num_samples=len(train_ds),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"\n[DATA] Train : {n_train} samples")
    print(f"[DATA] Val   : {n_val} samples")
    print(f"[DATA] Test  : {len(test_ds)} samples")
    print(f"[DATA] Classes: {full_train.class_to_idx}")

    return train_loader, val_loader, test_loader, full_train.class_to_idx


# ─────────────────────────────────────────────
#  CNN MODEL
# ─────────────────────────────────────────────
class FERNet(nn.Module):
    def __init__(self, filters=64, dropout=0.4, dense_units=512, num_classes=7):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout)
            )

        self.block1 = conv_block(1,          filters)
        self.block2 = conv_block(filters,    filters * 2)
        self.block3 = conv_block(filters*2,  filters * 4)

        feat_size = (IMG_SIZE // 8) ** 2 * filters * 4

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_size, dense_units),
            nn.BatchNorm1d(dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(dense_units, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


# ─────────────────────────────────────────────
#  FOCAL LOSS
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss: tập trung vào các sample khó (hard examples).
    Giảm loss của sample dễ (happy, surprise) → ép model học
    các class khó (fear, sad) tốt hơn → cải thiện recall.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    gamma=2: focus mạnh vào hard examples
    """
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.1):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt     = torch.exp(-ce)
        focal  = (1 - pt) ** self.gamma * ce
        return focal.mean()


# ─────────────────────────────────────────────
#  CLASS WEIGHTS (loss weighting)
# ─────────────────────────────────────────────
def get_class_weights(train_loader, device):
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)
    classes = np.unique(all_labels)
    weights = compute_class_weight('balanced', classes=classes, y=all_labels)

    print("\n[CLASS WEIGHTS]")
    for i, w in enumerate(weights):
        print(f"  {EMOTIONS[i]:10s}: {w:.3f}")

    return torch.FloatTensor(weights).to(device)


# ─────────────────────────────────────────────
#  HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
def run_tuner(train_loader, val_loader, device, n_trials=10):
    try:
        import optuna
    except ImportError:
        print("[TUNER] optuna not installed: pip install optuna")
        return None

    def objective(trial):
        f  = trial.suggest_categorical('filters',      [32, 64, 128])
        d  = trial.suggest_float('dropout',            0.2, 0.5, step=0.1)
        du = trial.suggest_categorical('dense_units',  [256, 512, 1024])
        lr = trial.suggest_float('learning_rate',      1e-4, 1e-2, log=True)

        model = FERNet(filters=f, dropout=d, dense_units=du).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(3):   # 3 epochs mỗi trial
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()

        # Val accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).argmax(1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
        return correct / total

    print(f"\n[TUNER] Starting Optuna search — {n_trials} trials...")
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n[TUNER] Best hyperparameters:")
    for k, v in best.items():
        print(f"  {k:15s}: {v}")
    return best


# ─────────────────────────────────────────────
#  LOGGER
# ─────────────────────────────────────────────
class TrainingLogger:
    def __init__(self, log_path, append=False):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        self.history  = {k: [] for k in
                         ['epoch','train_loss','train_acc','val_loss','val_acc',
                          'val_f1_weighted','lr','epoch_time']}
        if append and os.path.exists(log_path):
            # Đọc history cũ để plot đầy đủ cả trước và sau resume
            with open(log_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for k in self.history:
                        try:
                            self.history[k].append(float(row[k]))
                        except (KeyError, ValueError):
                            pass
            print(f"[LOG] Resumed log: {log_path} ({len(self.history['epoch'])} epochs loaded)")
        else:
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(list(self.history.keys()))
            print(f"[LOG] Training log: {log_path}")

    def log(self, **kwargs):
        for k, v in kwargs.items():
            self.history[k].append(v)
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([kwargs.get(k, '') for k in self.history.keys()])

    def print_epoch(self, epoch, total_epochs, **kwargs):
        train_loss = kwargs.get('train_loss', 0)
        train_acc  = kwargs.get('train_acc',  0)
        val_loss   = kwargs.get('val_loss',   0)
        val_acc    = kwargs.get('val_acc',    0)
        val_f1     = kwargs.get('val_f1_weighted', 0)
        lr         = kwargs.get('lr',         0)
        t          = kwargs.get('epoch_time', 0)

        bar_len = 20
        filled  = int(bar_len * epoch / total_epochs)
        bar     = '█' * filled + '░' * (bar_len - filled)

        print(f"\nEpoch {epoch:3d}/{total_epochs} [{bar}]  {t:.1f}s")
        print(f"  Train │ loss: {train_loss:.4f}  acc: {train_acc*100:.2f}%")
        print(f"  Val   │ loss: {val_loss:.4f}  acc: {val_acc*100:.2f}%  "
              f"F1(w): {val_f1:.4f}")
        print(f"  LR    │ {lr:.6f}")


# ─────────────────────────────────────────────
#  TRAIN ONE EPOCH
# ─────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0

    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # FORWARD (Lab 1: linear_forward)
        logits = model(X)
        loss   = criterion(logits, y)

        # BACKWARD (Lab 1: L_model_backward)
        loss.backward()

        # Gradient clipping — tránh exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # UPDATE (Lab 1: update_parameters)
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds       = logits.argmax(1)
        correct    += (preds == y).sum().item()
        total      += y.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────
#  VALIDATE
# ─────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_preds = []
    all_labels = []

    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        loss   = criterion(logits, y)

        total_loss += loss.item() * X.size(0)
        preds       = logits.argmax(1)
        correct    += (preds == y).sum().item()
        total      += y.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / total
    acc      = correct / total
    f1       = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return avg_loss, acc, f1, all_preds, all_labels


# ─────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────
def export_onnx_inline(model, device, path):
    """Export ONNX ngay sau khi save best checkpoint."""
    model.eval()
    dummy = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(device)
    torch.onnx.export(
        model, dummy, path,
        input_names=['face'], output_names=['logits'],
        dynamic_axes={'face': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=17
    )
    print(f"  → ONNX exported: {path}")


def train(model, train_loader, val_loader, device, args, logger,
          class_weights=None, start_epoch=1, best_val_acc=0.0,
          optimizer_state=None):
    # Focal Loss: cải thiện recall cho fear/sad
    criterion = FocalLoss(gamma=2.0, weight=class_weights, label_smoothing=0.1)
    print(f"[TRAIN] Loss: FocalLoss(gamma=2.0, label_smoothing=0.1)")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        print(f"[RESUME] Optimizer state loaded")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=False)

    patience_count = 0
    best_path      = os.path.join(OUTPUT_DIR, 'best.pt')
    onnx_path      = os.path.join(OUTPUT_DIR, 'emotion_model.onnx')

    mode = "RESUME" if start_epoch > 1 else "START"
    print(f"\n[TRAIN] {mode} — epoch {start_epoch} → {start_epoch + args.epochs - 1} on {device}")
    print(f"[TRAIN] Best val_acc so far: {best_val_acc*100:.2f}%")
    print(f"{'─'*70}")
    print(f"{'Epoch':>6} │ {'TrainLoss':>9} {'TrainAcc':>9} │ "
          f"{'ValLoss':>8} {'ValAcc':>8} {'F1(w)':>7} │ {'LR':>10} │ {'Time':>6}")
    print(f"{'─'*70}")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        elapsed    = time.time() - t0

        is_best = val_acc > best_val_acc
        marker  = ' ◄ BEST' if is_best else ''
        print(f"{epoch:6d} │ {train_loss:9.4f} {train_acc*100:8.2f}% │ "
              f"{val_loss:8.4f} {val_acc*100:8.2f}% {val_f1:7.4f} │ "
              f"{current_lr:10.6f} │ {elapsed:5.1f}s{marker}")

        # CSV log
        logger.log(
            epoch=epoch, train_loss=round(train_loss, 4),
            train_acc=round(train_acc, 4), val_loss=round(val_loss, 4),
            val_acc=round(val_acc, 4), val_f1_weighted=round(val_f1, 4),
            lr=round(current_lr, 6), epoch_time=round(elapsed, 1)
        )

        # Save best + export ONNX ngay lập tức
        if is_best:
            best_val_acc = val_acc
            torch.save({
                'epoch'          : epoch,
                'model_state'    : model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc'        : val_acc,
                'val_f1'         : val_f1,
                'args'           : vars(args)
            }, best_path)
            export_onnx_inline(model, device, onnx_path)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\n[TRAIN] Early stopping at epoch {epoch} "
                      f"(patience={args.patience})")
                break

    print(f"{'─'*70}")
    print(f"[TRAIN] Best val_acc: {best_val_acc*100:.2f}%  →  {best_path}")
    return best_val_acc


# ─────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────
def evaluate(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_f1, y_pred, y_true = validate(
        model, test_loader, criterion, device)

    print(f"\n{'='*55}")
    print(f"  TEST RESULTS")
    print(f"{'='*55}")
    print(f"  Loss     : {test_loss:.4f}")
    print(f"  Accuracy : {test_acc*100:.2f}%")
    print(f"  F1 (w)   : {test_f1:.4f}")
    print(f"{'='*55}")
    print(f"\n[CLASSIFICATION REPORT]")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS, zero_division=0))

    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print(f"[PER-CLASS ACCURACY]")
    for i, em in enumerate(EMOTIONS):
        bar = '█' * int(per_class_acc[i] * 20)
        print(f"  {em:10s}: {per_class_acc[i]*100:5.1f}%  {bar}")

    return test_acc, test_f1, cm, y_true, y_pred


# export_onnx_inline được định nghĩa trong hàm train() ở trên


# ─────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────
def save_plots(logger, cm, save_dir):
    h = logger.history

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(h['train_loss'], label='Train', color='royalblue')
    axes[0].plot(h['val_loss'],   label='Val',   color='tomato')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)

    axes[1].plot([v*100 for v in h['train_acc']], label='Train', color='royalblue')
    axes[1].plot([v*100 for v in h['val_acc']],   label='Val',   color='tomato')
    axes[1].set_title('Accuracy (%)'); axes[1].legend(); axes[1].grid(True)

    axes[2].plot(h['val_f1_weighted'], label='Val F1 (weighted)', color='seagreen')
    axes[2].set_title('Weighted F1'); axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    curve_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"[PLOT] Training curves: {curve_path}")

    # Confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[PLOT] Confusion matrix: {cm_path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    train_dir = os.path.join(args.data, 'train')
    test_dir  = os.path.join(args.data, 'test')

    for d in [train_dir, test_dir]:
        if not os.path.exists(d):
            print(f"[ERROR] Not found: {d}")
            sys.exit(1)

    os.makedirs(OUTPUT_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"\n{'='*55}")
    print(f"  Face Emotion Recognition — Training")
    print(f"  Started: {ts}")
    print(f"{'='*55}")
    print(f"  Dataset : {args.data}")
    print(f"  Epochs  : {args.epochs}  |  Batch: {args.batch}")
    print(f"  Tune    : {'No (--no-tune)' if args.no_tune else f'Optuna {args.tune_trials} trials'}")
    print(f"{'='*55}")

    device = check_device()

    # Data
    train_loader, val_loader, test_loader, class_to_idx = build_loaders(
        train_dir, test_dir, args.batch)

    # Save class indices
    with open(os.path.join(OUTPUT_DIR, 'class_indices.json'), 'w') as f:
        json.dump(class_to_idx, f, indent=2)

    # ── Resume từ checkpoint ──────────────────────
    start_epoch    = 1
    best_val_acc   = 0.0
    optimizer_state = None
    filters, dropout, dense, lr = args.filters, args.dropout, args.dense, args.lr

    if args.resume:
        if not os.path.exists(args.resume):
            print(f"[ERROR] Checkpoint not found: {args.resume}")
            sys.exit(1)
        ckpt = torch.load(args.resume, map_location=device)
        filters     = ckpt['args'].get('filters',  filters)
        dropout     = ckpt['args'].get('dropout',  dropout)
        dense       = ckpt['args'].get('dense',    dense)
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt['val_acc']
        optimizer_state = ckpt.get('optimizer_state')
        print(f"\n[RESUME] Checkpoint: {args.resume}")
        print(f"[RESUME] Epoch {ckpt['epoch']}  val_acc={best_val_acc*100:.2f}%")
        print(f"[RESUME] Tiếp tục từ epoch {start_epoch} thêm {args.epochs} epochs")
    else:
        # Hyperparameter tuning (chỉ khi train mới)
        if not args.no_tune:
            best_params = run_tuner(train_loader, val_loader, device, args.tune_trials)
            if best_params:
                filters = best_params.get('filters', filters)
                dropout = best_params.get('dropout', dropout)
                dense   = best_params.get('dense_units', dense)
                lr      = best_params.get('learning_rate', lr)

    print(f"\n[MODEL] CNN: filters={filters}, dropout={dropout:.2f}, "
          f"dense={dense}, lr={lr:.5f}")

    # Build model
    model = FERNet(filters=filters, dropout=dropout,
                   dense_units=dense, num_classes=len(EMOTIONS)).to(device)

    # Load weights nếu resume
    if args.resume:
        model.load_state_dict(ckpt['model_state'])
        print(f"[RESUME] Weights loaded ✓")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Trainable parameters: {total_params:,}")

    # Class weights cho Focal Loss
    class_weights = get_class_weights(train_loader, device)

    # Logger — append nếu resume, tạo mới nếu train fresh
    append_mode = args.resume is not None
    logger = TrainingLogger(LOG_FILE, append=append_mode)

    # Train
    t_start = time.time()
    best_val_acc = train(
        model, train_loader, val_loader, device, args, logger,
        class_weights=class_weights,
        start_epoch=start_epoch,
        best_val_acc=best_val_acc,
        optimizer_state=optimizer_state
    )
    total_time = time.time() - t_start
    print(f"\n[TRAIN] Total time: {total_time/60:.1f} minutes")

    # Load best weights để evaluate
    best_path  = os.path.join(OUTPUT_DIR, 'best.pt')
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    print(f"[LOAD] Best checkpoint: epoch {checkpoint['epoch']}, "
          f"val_acc={checkpoint['val_acc']*100:.2f}%")

    # Evaluate
    test_acc, test_f1, cm, y_true, y_pred = evaluate(model, test_loader, device)

    # Plots
    save_plots(logger, cm, RESULTS_DIR)

    onnx_path = os.path.join(OUTPUT_DIR, 'emotion_model.onnx')
    print(f"\n{'='*55}")
    print(f"  SUMMARY")
    print(f"{'='*55}")
    print(f"  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Test F1 (w)  : {test_f1:.4f}")
    print(f"  Training time : {total_time/60:.1f} min")
    print(f"  Model saved   : {best_path}")
    print(f"  ONNX exported : {onnx_path}")
    print(f"  Log file      : {LOG_FILE}")
    print(f"  Plots         : {RESULTS_DIR}/")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
