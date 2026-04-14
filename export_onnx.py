"""
Export FERNet PyTorch → ONNX
Usage:
    python export_onnx.py --checkpoint models/best.pt
    python export_onnx.py --checkpoint models/best.pt --filters 64 --dense 512
"""

import argparse
import os
import sys
import json

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# ── Copy FERNet từ train.py (để không phụ thuộc import) ──
IMG_SIZE  = 48
EMOTIONS  = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


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

        self.block1 = conv_block(1,         filters)
        self.block2 = conv_block(filters,   filters * 2)
        self.block3 = conv_block(filters*2, filters * 4)

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to .pt checkpoint (vd: models/best.pt)')
    p.add_argument('--output', type=str, default='models/emotion_model.onnx')
    p.add_argument('--filters', type=int, default=64)
    p.add_argument('--dense',   type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Không tìm thấy: {args.checkpoint}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    device = torch.device('cpu')  # export trên CPU để ONNX portable hơn

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Checkpoint có thể là dict hoặc state_dict thẳng
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        state_dict = ckpt['model_state']
        val_acc    = ckpt.get('val_acc', 0)
        epoch      = ckpt.get('epoch', '?')
        print(f"[INFO] Checkpoint: epoch={epoch}, val_acc={val_acc*100:.2f}%")
    else:
        state_dict = ckpt
        print("[INFO] Checkpoint: raw state_dict")

    # Khởi tạo model
    model = FERNet(filters=args.filters, dense_units=args.dense)
    model.load_state_dict(state_dict)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model params: {params:,}")

    # Dummy input: batch=1, channel=1 (grayscale), 48×48
    # Normalize [-1, 1] như training (mean=0.5, std=0.5)
    dummy = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)

    # Kiểm tra forward pass
    with torch.no_grad():
        out = model(dummy)
    print(f"[INFO] Forward pass OK — output shape: {out.shape}")

    # Export ONNX
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=['face'],
        output_names=['logits'],
        dynamic_axes={
            'face':   {0: 'batch'},
            'logits': {0: 'batch'}
        },
        opset_version=17,
        do_constant_folding=True
    )
    print(f"[OK] ONNX saved: {args.output}")

    # Verify ONNX
    try:
        import onnxruntime as ort
        from scipy.special import softmax

        sess = ort.InferenceSession(args.output,
                                    providers=['CPUExecutionProvider'])
        inp_name = sess.get_inputs()[0].name

        # Chạy thử với dummy
        test_input = dummy.numpy()
        logits = sess.run(None, {inp_name: test_input})[0][0]
        probs  = softmax(logits)

        print(f"\n[VERIFY] ONNX inference OK")
        print(f"  Input  : {test_input.shape}  (NCHW, float32)")
        print(f"  Logits : {logits.round(3)}")
        print(f"\n  Emotion probabilities (dummy input):")
        for em, p in zip(EMOTIONS, probs):
            print(f"  {em:10s}: {p*100:5.1f}%")

    except ImportError:
        print("[WARN] onnxruntime không có — bỏ qua verify")

    # Lưu class_indices.json
    class_indices_path = os.path.join(
        os.path.dirname(args.output), 'class_indices.json')
    class_indices = {em: i for i, em in enumerate(EMOTIONS)}
    with open(class_indices_path, 'w') as f:
        json.dump(class_indices, f, indent=2)
    print(f"[OK] class_indices.json saved: {class_indices_path}")

    print("\n[DONE] Buoc tiep theo:")
    print("  1. Train audio: python train_audio.py --data D:/ravdess --epochs 100")
    print(f"  2. Demo       : python demo_multimodal.py --face-model {args.output}")


if __name__ == '__main__':
    main()
