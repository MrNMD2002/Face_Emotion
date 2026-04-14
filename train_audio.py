"""
Audio Emotion Recognition - Training Script
============================================
Dataset : RAVDESS (Actor_01 ... Actor_24, moi actor co file .wav)
Model   : CNN + BiLSTM tren MFCC features
Output  : models/audio_best.pt  +  models/audio_model.onnx

Usage:
    # Colab / Linux server (GPU)
    python train_audio.py --data /path/to/ravdess --epochs 100 --batch 64

    # Resume tu checkpoint
    python train_audio.py --data /path/to/ravdess --resume models/audio_best.pt

    # Windows local (khong GPU)
    python train_audio.py --data D:/ravdess --epochs 100 --batch 32 --workers 0
"""

import os
import sys
import json
import argparse
import time
import csv
import glob
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from audio_stream import (AudioEmotionNet, extract_mfcc, load_audio_file,
                          EMOTIONS, RAVDESS_MAP, N_MFCC, MAX_FRAMES)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
OUTPUT_DIR  = 'models'
RESULTS_DIR = 'results'
LOG_FILE    = 'results/audio_training_log.csv'


def parse_args():
    p = argparse.ArgumentParser(description='Train Audio Emotion Model on RAVDESS')
    p.add_argument('--data',     type=str, required=True,
                   help='Root RAVDESS dir chua Actor_01...Actor_24')
    p.add_argument('--epochs',   type=int,   default=100)
    p.add_argument('--batch',    type=int,   default=64)
    p.add_argument('--lr',       type=float, default=1e-3)
    p.add_argument('--dropout',  type=float, default=0.3)
    p.add_argument('--patience', type=int,   default=15)
    p.add_argument('--workers',  type=int,   default=2,
                   help='DataLoader workers: 0=Windows, 2=Linux/Colab')
    p.add_argument('--resume',   type=str,   default=None,
                   help='Path checkpoint .pt de train tiep')
    p.add_argument('--seed',     type=int,   default=42)
    return p.parse_args()


# ─────────────────────────────────────────────
#  RAVDESS DATASET
# ─────────────────────────────────────────────
class RAVDESSDataset(Dataset):
    """
    Load RAVDESS .wav files va extract MFCC.

    Filename format: 03-01-{emotion}-{intensity}-{stmt}-{rep}-{actor}.wav
    emotion codes : 01=neutral, 02=calm->neutral, 03=happy, 04=sad
                    05=angry,   06=fear,            07=disgust, 08=surprise
    """
    def __init__(self, file_list, augment=False):
        self.files        = file_list
        self.augment      = augment
        self.class_to_idx = {e: i for i, e in enumerate(EMOTIONS)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        parts = os.path.basename(path).replace('.wav', '').split('-')
        emotion_str = RAVDESS_MAP.get(parts[2], 'neutral')
        label       = self.class_to_idx[emotion_str]

        audio = load_audio_file(path)
        if self.augment:
            audio = self._augment(audio)

        mfcc = extract_mfcc(audio)          # [1, 1, N_MFCC, T]
        return torch.FloatTensor(mfcc[0]), torch.LongTensor([label])[0]

    def _augment(self, audio):
        import librosa
        # Gaussian noise
        if np.random.random() < 0.4:
            audio = audio + np.random.randn(len(audio)) * 0.005
        # Time stretch
        if np.random.random() < 0.3:
            rate  = np.random.uniform(0.85, 1.15)
            try:
                audio = librosa.effects.time_stretch(audio, rate=rate)
            except Exception:
                pass
        # Pitch shift
        if np.random.random() < 0.3:
            steps = int(np.random.uniform(-3, 3))
            try:
                audio = librosa.effects.pitch_shift(audio, sr=16000, n_steps=steps)
            except Exception:
                pass
        return audio


def load_ravdess(data_dir):
    """Quet tat ca .wav trong RAVDESS, loc file hop le, in distribution."""
    files = glob.glob(os.path.join(data_dir, '**', '*.wav'), recursive=True)

    valid, label_count = [], {e: 0 for e in EMOTIONS}
    for f in files:
        parts = os.path.basename(f).replace('.wav', '').split('-')
        if len(parts) >= 3 and parts[2] in RAVDESS_MAP:
            label_count[RAVDESS_MAP[parts[2]]] += 1
            valid.append(f)

    print(f"\n[DATA] Found {len(valid)} valid audio files")
    print("[DATA] Distribution:")
    for em, cnt in label_count.items():
        print(f"  {em:10s}: {cnt:4d}")

    if len(valid) == 0:
        raise RuntimeError(f"Khong tim thay .wav hop le trong: {data_dir}")

    return valid


def build_loaders(files, batch_size, workers, seed=42):
    """Split 70/15/15, tao DataLoader."""
    train_files, test_files = train_test_split(
        files, test_size=0.15, random_state=seed)
    train_files, val_files  = train_test_split(
        train_files, test_size=0.15, random_state=seed)

    train_ds = RAVDESSDataset(train_files, augment=True)
    val_ds   = RAVDESSDataset(val_files,   augment=False)
    test_ds  = RAVDESSDataset(test_files,  augment=False)

    pin = torch.cuda.is_available()
    kw  = dict(num_workers=workers, pin_memory=pin, persistent_workers=(workers > 0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw)

    print(f"[DATA] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
#  CLASS WEIGHTS
# ─────────────────────────────────────────────
def get_class_weights(train_loader, device):
    labels = []
    for _, y in train_loader:
        labels.extend(y.numpy())
    labels  = np.array(labels)
    weights = compute_class_weight('balanced',
                                   classes=np.unique(labels), y=labels)
    print("\n[CLASS WEIGHTS]")
    for i, w in enumerate(weights):
        print(f"  {EMOTIONS[i]:10s}: {w:.3f}")
    return torch.FloatTensor(weights).to(device)


# ─────────────────────────────────────────────
#  FOCAL LOSS
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss giam tam quan trong cua easy samples,
    ep model tap trung vao hard samples (fear, disgust).
    """
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.1):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce    = nn.functional.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction='none')
        pt    = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()


# ─────────────────────────────────────────────
#  TRAIN / VALIDATE
# ─────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0

    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    for X, y in loader:
        X, y   = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        loss   = criterion(logits, y)

        total_loss += loss.item() * X.size(0)
        preds       = logits.argmax(1)
        correct    += (preds == y).sum().item()
        total      += y.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    acc = correct / total
    f1  = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return total_loss / total, acc, f1, all_preds, all_labels


# ─────────────────────────────────────────────
#  EXPORT ONNX
# ─────────────────────────────────────────────
def export_onnx(model, device, path):
    model.eval()
    dummy = torch.randn(1, 1, N_MFCC, MAX_FRAMES).to(device)
    torch.onnx.export(
        model, dummy, path,
        input_names=['mfcc'],
        output_names=['logits'],
        dynamic_axes={'mfcc': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=17,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(path) / 1024**2
    print(f"  [ONNX] Saved: {path}  ({size_mb:.1f} MB)")


# ─────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────
def run_training(model, train_loader, val_loader, device, args,
                 class_weights=None, start_epoch=1,
                 best_val_acc=0.0, optimizer_state=None):

    criterion = FocalLoss(gamma=2.0, weight=class_weights, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    scheduler      = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, verbose=False)
    patience_count = 0
    best_path      = os.path.join(OUTPUT_DIR, 'audio_best.pt')
    onnx_path      = os.path.join(OUTPUT_DIR, 'audio_model.onnx')

    # CSV log
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_mode = 'a' if args.resume else 'w'
    with open(LOG_FILE, log_mode, newline='') as f:
        if log_mode == 'w':
            csv.writer(f).writerow(
                ['epoch', 'train_loss', 'train_acc',
                 'val_loss', 'val_acc', 'val_f1', 'lr', 'time_s'])

    end_epoch = start_epoch + args.epochs - 1
    print(f"\n[TRAIN] Epochs {start_epoch} -> {end_epoch}  |  device={device}")
    print(f"{'Epoch':>6} | {'TrLoss':>8} {'TrAcc':>7} | "
          f"{'VaLoss':>8} {'VaAcc':>7} {'F1':>6} | {'LR':>9} | {'t(s)':>5}")
    print("-" * 70)

    for epoch in range(start_epoch, end_epoch + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1, _, _ = validate(model, val_loader, criterion, device)

        scheduler.step(va_acc)
        lr      = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        is_best = va_acc > best_val_acc
        tag     = ' *' if is_best else ''
        print(f"{epoch:6d} | {tr_loss:8.4f} {tr_acc*100:6.2f}% | "
              f"{va_loss:8.4f} {va_acc*100:6.2f}% {va_f1:6.4f} | "
              f"{lr:9.6f} | {elapsed:5.1f}{tag}")

        with open(LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch,
                round(tr_loss, 4), round(tr_acc, 4),
                round(va_loss, 4), round(va_acc, 4),
                round(va_f1, 4),   round(lr, 6),
                round(elapsed, 1)
            ])

        if is_best:
            best_val_acc = va_acc
            torch.save({
                'epoch'          : epoch,
                'model_state'    : model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc'        : va_acc,
                'val_f1'         : va_f1,
            }, best_path)
            export_onnx(model, device, onnx_path)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\n[TRAIN] Early stopping at epoch {epoch} "
                      f"(patience={args.patience})")
                break

    print("-" * 70)
    print(f"[TRAIN] Best val_acc = {best_val_acc*100:.2f}%  ->  {best_path}")
    return best_val_acc


# ─────────────────────────────────────────────
#  EVALUATE ON TEST SET
# ─────────────────────────────────────────────
def evaluate(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    _, acc, f1, y_pred, y_true = validate(model, test_loader, criterion, device)

    print(f"\n{'='*55}")
    print(f"  AUDIO TEST RESULTS")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  F1 (w)   : {f1:.4f}")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred,
                                target_names=EMOTIONS, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title(f'Audio Model - Confusion Matrix (acc={acc*100:.1f}%)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, 'audio_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved: {cm_path}")

    # Per-class accuracy
    per_class = cm.diagonal() / cm.sum(axis=1)
    print("\n[PER-CLASS ACCURACY]")
    for i, em in enumerate(EMOTIONS):
        print(f"  {em:10s}: {per_class[i]*100:5.1f}%")

    return acc, f1


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(OUTPUT_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Audio Emotion Recognition - Training")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Dataset : {args.data}")
    print(f"  Epochs  : {args.epochs} | Batch: {args.batch} | Workers: {args.workers}")
    print(f"{'='*55}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"[GPU] {gpu.name}  {gpu.total_memory/1024**3:.1f} GB VRAM")
    else:
        print("[WARNING] No GPU - training on CPU (slow)")

    # Data
    files = load_ravdess(args.data)
    train_loader, val_loader, test_loader = build_loaders(
        files, args.batch, args.workers, seed=args.seed)

    # Model
    model = AudioEmotionNet(
        n_mfcc=N_MFCC,
        num_classes=len(EMOTIONS),
        dropout=args.dropout
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Params: {params:,}")

    # Resume
    start_epoch     = 1
    best_val_acc    = 0.0
    optimizer_state = None

    if args.resume:
        if not os.path.exists(args.resume):
            print(f"[ERROR] Checkpoint not found: {args.resume}")
            sys.exit(1)
        ckpt            = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        start_epoch     = ckpt['epoch'] + 1
        best_val_acc    = ckpt['val_acc']
        optimizer_state = ckpt.get('optimizer_state')
        print(f"[RESUME] From epoch {ckpt['epoch']}, val_acc={best_val_acc*100:.2f}%")

    # Class weights
    class_weights = get_class_weights(train_loader, device)

    # Train
    t0 = time.time()
    run_training(model, train_loader, val_loader, device, args,
                 class_weights=class_weights,
                 start_epoch=start_epoch,
                 best_val_acc=best_val_acc,
                 optimizer_state=optimizer_state)
    print(f"\n[TRAIN] Total time: {(time.time()-t0)/60:.1f} min")

    # Evaluate best model on test set
    ckpt = torch.load(os.path.join(OUTPUT_DIR, 'audio_best.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state'])
    evaluate(model, test_loader, device)

    # class_indices.json — ghi de neu da co (dung chung voi face model)
    class_indices = {e: i for i, e in enumerate(EMOTIONS)}
    ci_path = os.path.join(OUTPUT_DIR, 'class_indices.json')
    with open(ci_path, 'w') as f:
        json.dump(class_indices, f, indent=2)
    print(f"[OK] class_indices.json: {ci_path}")

    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"  Checkpoint : {OUTPUT_DIR}/audio_best.pt")
    print(f"  ONNX model : {OUTPUT_DIR}/audio_model.onnx")
    print(f"  Log        : {LOG_FILE}")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
