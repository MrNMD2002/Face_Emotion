"""
Late Fusion — Face + Audio
===========================
Kết hợp 2 probability vectors từ face và audio model.
Tìm weight tối ưu trên validation set bằng scipy.
"""

import os
import json
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# ─────────────────────────────────────────────
#  LATE FUSION
# ─────────────────────────────────────────────
class LateFusion:
    """
    Ket hop face_probs + audio_probs.

    2 che do:
      - fixed      : w co dinh (mac dinh 0.5/0.5)
      - confidence : w tu dong theo do tu tin tung frame
    """
    def __init__(self, w_face=0.5, w_audio=0.5, mode='confidence'):
        self.w_face  = w_face
        self.w_audio = w_audio
        self.mode    = mode   # 'fixed' hoac 'confidence'

    # ── Confidence-Based Fusion ───────────────────────────────────────
    def fuse(self, face_probs, audio_probs):
        """
        Args:
            face_probs  : np.array [7]
            audio_probs : np.array [7]
        Returns:
            (emotion, confidence, fused_probs)
        """
        if self.mode == 'confidence':
            w_face, w_audio = self._confidence_weights(face_probs, audio_probs)
        else:
            w_face, w_audio = self.w_face, self.w_audio

        fused      = w_face * face_probs + w_audio * audio_probs
        idx        = int(np.argmax(fused))
        emotion    = EMOTIONS[idx]
        confidence = float(fused[idx])
        return emotion, confidence, fused

    def _confidence_weights(self, face_probs, audio_probs):
        """
        Tinh w tu dong theo do tu tin cua tung model.

        Nguyen ly:
          conf_face  = max(face_probs)   <- model tu tin bao nhieu
          conf_audio = max(audio_probs)

          w_face  = conf_face  / (conf_face + conf_audio)
          w_audio = conf_audio / (conf_face + conf_audio)

        Vi du:
          face  -> HAPPY 90%  : conf = 0.9
          audio -> HAPPY 30%  : conf = 0.3
          w_face  = 0.9/1.2 = 0.75  (tin face hon vi face tu tin hon)
          w_audio = 0.3/1.2 = 0.25
        """
        conf_face  = float(np.max(face_probs))
        conf_audio = float(np.max(audio_probs))
        total      = conf_face + conf_audio + 1e-8
        return conf_face / total, conf_audio / total

    def fuse_batch(self, face_probs_all, audio_probs_all):
        """Fuse batch [N, 7]."""
        if self.mode == 'confidence':
            # Tinh confidence tung sample
            conf_face  = np.max(face_probs_all,  axis=1, keepdims=True)
            conf_audio = np.max(audio_probs_all, axis=1, keepdims=True)
            total      = conf_face + conf_audio + 1e-8
            w_face     = conf_face  / total
            w_audio    = conf_audio / total
            return w_face * face_probs_all + w_audio * audio_probs_all
        return self.w_face * face_probs_all + self.w_audio * audio_probs_all

    def find_optimal_weights(self, face_probs_all, audio_probs_all, y_true,
                             metric='accuracy'):
        """
        Tìm w_face tối ưu trên validation set.
        w_audio = 1 - w_face

        Args:
            face_probs_all  : [N, 7]
            audio_probs_all : [N, 7]
            y_true          : [N] int labels
            metric          : 'accuracy' hoặc 'f1'
        """
        def objective(w):
            fused  = w * face_probs_all + (1 - w) * audio_probs_all
            y_pred = np.argmax(fused, axis=1)
            if metric == 'f1':
                score = f1_score(y_true, y_pred, average='weighted',
                                 zero_division=0)
            else:
                score = accuracy_score(y_true, y_pred)
            return -score   # minimize → negate

        result       = minimize_scalar(objective, bounds=(0.0, 1.0),
                                       method='bounded')
        self.w_face  = float(result.x)
        self.w_audio = 1.0 - self.w_face

        best_score = -result.fun
        print(f"\n[FUSION] Optimal weights found:")
        print(f"  w_face  = {self.w_face:.3f}")
        print(f"  w_audio = {self.w_audio:.3f}")
        print(f"  Best {metric}: {best_score*100:.2f}%")
        return self.w_face, self.w_audio

    def save(self, path='models/fusion_weights.json'):
        with open(path, 'w') as f:
            json.dump({'w_face': self.w_face, 'w_audio': self.w_audio}, f,
                      indent=2)
        print(f"[FUSION] Weights saved: {path}")

    def load(self, path='models/fusion_weights.json'):
        with open(path) as f:
            d = json.load(f)
        self.w_face  = d['w_face']
        self.w_audio = d['w_audio']
        print(f"[FUSION] Loaded: w_face={self.w_face:.3f}, "
              f"w_audio={self.w_audio:.3f}")


# ─────────────────────────────────────────────
#  EVALUATOR
# ─────────────────────────────────────────────
class FusionEvaluator:
    """
    So sánh Face only / Audio only / Fused trên cùng test set.
    """
    def __init__(self, fusion: LateFusion):
        self.fusion = fusion

    def compare(self, face_probs_all, audio_probs_all, y_true):
        """
        In bảng so sánh và vẽ confusion matrix 3 luồng.
        """
        y_face  = np.argmax(face_probs_all,  axis=1)
        y_audio = np.argmax(audio_probs_all, axis=1)
        fused   = self.fusion.fuse_batch(face_probs_all, audio_probs_all)
        y_fused = np.argmax(fused, axis=1)

        print(f"\n{'='*60}")
        print(f"  FUSION EVALUATION")
        print(f"{'='*60}")
        print(f"  {'Stream':15s} {'Accuracy':>10} {'F1 (w)':>10}")
        print(f"  {'─'*40}")

        for name, y_pred in [('Face only',  y_face),
                              ('Audio only', y_audio),
                              ('Fused',      y_fused)]:
            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            tag = ' ◄' if name == 'Fused' else ''
            print(f"  {name:15s} {acc*100:9.2f}% {f1:10.4f}{tag}")

        print(f"\n[PER-CLASS RECALL COMPARISON]")
        print(f"  {'Class':10s} │ {'Face':>6} {'Audio':>6} {'Fused':>6} │ {'Δ':>6}")
        print(f"  {'─'*50}")

        from sklearn.metrics import recall_score
        r_face  = recall_score(y_true, y_face,  average=None, zero_division=0)
        r_audio = recall_score(y_true, y_audio, average=None, zero_division=0)
        r_fused = recall_score(y_true, y_fused, average=None, zero_division=0)

        for i, em in enumerate(EMOTIONS):
            delta = r_fused[i] - max(r_face[i], r_audio[i])
            arrow = '↑' if delta > 0 else ('↓' if delta < 0 else '→')
            print(f"  {em:10s} │ {r_face[i]:6.2f} {r_audio[i]:6.2f} "
                  f"{r_fused[i]:6.2f} │ {arrow} {abs(delta):.2f}")

        return y_face, y_audio, y_fused

    def plot_confusion_matrices(self, face_probs_all, audio_probs_all,
                                y_true, save_dir='results'):
        import matplotlib.pyplot as plt
        import seaborn as sns

        y_face  = np.argmax(face_probs_all,  axis=1)
        y_audio = np.argmax(audio_probs_all, axis=1)
        fused   = self.fusion.fuse_batch(face_probs_all, audio_probs_all)
        y_fused = np.argmax(fused, axis=1)

        fig, axes = plt.subplots(1, 3, figsize=(24, 7))

        for ax, y_pred, title, cmap in zip(
            axes,
            [y_face, y_audio, y_fused],
            ['Face Only', 'Audio Only', 'Fused'],
            ['Blues', 'Oranges', 'Greens']
        ):
            cm  = confusion_matrix(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                        xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax)
            ax.set_title(f'{title}\nAcc={acc*100:.1f}%  F1={f1:.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

        plt.tight_layout()
        path = os.path.join(save_dir, 'fusion_comparison.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[PLOT] Saved: {path}")

    def plot_weight_sensitivity(self, face_probs_all, audio_probs_all,
                                y_true, save_dir='results'):
        """Vẽ accuracy theo từng giá trị w_face từ 0 → 1."""
        import matplotlib.pyplot as plt

        weights    = np.arange(0, 1.01, 0.05)
        accuracies = []
        f1_scores  = []

        for w in weights:
            fused  = w * face_probs_all + (1-w) * audio_probs_all
            y_pred = np.argmax(fused, axis=1)
            accuracies.append(accuracy_score(y_true, y_pred))
            f1_scores.append(f1_score(y_true, y_pred, average='weighted',
                                      zero_division=0))

        plt.figure(figsize=(10, 5))
        plt.plot(weights, [a*100 for a in accuracies],
                 'royalblue', label='Accuracy')
        plt.plot(weights, [f*100 for f in f1_scores],
                 'tomato',    label='F1 (weighted)')
        plt.axvline(self.fusion.w_face, color='green', linestyle='--',
                    label=f'Optimal w_face={self.fusion.w_face:.2f}')
        plt.xlabel('w_face  (w_audio = 1 - w_face)')
        plt.ylabel('%')
        plt.title('Fusion Weight Sensitivity')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        path = os.path.join(save_dir, 'fusion_weight_sensitivity.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[PLOT] Saved: {path}")
