"""
Multimodal Emotion Recognition — Real-time Demo
================================================
Kết hợp Face stream + Audio stream real-time.
- Face  : SCRFD detector + CNN emotion model (ONNX)
- Audio : Microphone + CNN+LSTM model (ONNX)
- Fusion: Weighted average (weight từ fusion_weights.json)

Usage:
    python demo_multimodal.py
    python demo_multimodal.py --face-model  models/emotion_model.onnx
                              --audio-model models/audio_model.onnx
                              --fusion      models/fusion_weights.json
Controls:
    q  — Quit
    s  — Screenshot
    f  — Toggle fusion / face-only mode
    k  — Toggle keypoints
    b  — Toggle confidence bars
"""

import cv2
import numpy as np
import argparse
import os
import sys
import json
import threading
import queue
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from face_emotion_pipeline import FaceEmotionPipeline, EMOTION_COLORS, EMOTIONS
from audio_stream import AudioPredictor, RealtimeAudioBuffer
from fusion import LateFusion

# ─────────────────────────────────────────────
#  DEFAULT PATHS
# ─────────────────────────────────────────────
DEFAULT_DETECTOR    = 'models/det_10g_fp16_dynamic.onnx'
DEFAULT_FACE_MODEL  = 'models/emotion_model.onnx'
DEFAULT_AUDIO_MODEL = 'models/audio_model.onnx'
DEFAULT_CLASSES     = 'models/class_indices.json'
DEFAULT_FUSION      = 'models/fusion_weights.json'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--detector',    default=DEFAULT_DETECTOR)
    p.add_argument('--face-model',  default=DEFAULT_FACE_MODEL)
    p.add_argument('--audio-model', default=DEFAULT_AUDIO_MODEL)
    p.add_argument('--classes',     default=DEFAULT_CLASSES)
    p.add_argument('--fusion',      default=DEFAULT_FUSION)
    p.add_argument('--cam',         type=int, default=0)
    p.add_argument('--det-thresh',  type=float, default=0.45)
    p.add_argument('--no-audio',    action='store_true',
                   help='Tắt audio stream, chỉ dùng face')
    return p.parse_args()


# ─────────────────────────────────────────────
#  DRAW HELPERS
# ─────────────────────────────────────────────
def draw_audio_label(frame, audio_result):
    """Label nhỏ gọn góc trái trên: AUDIO · HAPPY 78%"""
    em, conf, _ = audio_result
    color = EMOTION_COLORS.get(em, (200, 200, 200))
    label = f"AUDIO  {em.upper()} {conf*100:.0f}%"
    cv2.putText(frame, label, (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)


def draw_fused_result(frame, fused_em, fused_conf, fused_probs):
    """Vẽ kết quả fusion ở góc phải trên."""
    h, w = frame.shape[:2]
    color = EMOTION_COLORS.get(fused_em, (0, 255, 0))

    # Background panel
    panel_w, panel_h = 220, 35
    px = w - panel_w - 10
    py = 10
    cv2.rectangle(frame, (px-5, py-5), (px+panel_w, py+panel_h),
                  (30, 30, 30), -1)
    cv2.rectangle(frame, (px-5, py-5), (px+panel_w, py+panel_h),
                  color, 2)

    # Fused label
    label = f"FUSED: {fused_em.upper()} {fused_conf*100:.0f}%"
    cv2.putText(frame, label, (px, py+22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


class IPhoneWaveform:
    """
    iPhone Voice Memos style waveform.

    - Đặt ở bottom-center, không chiếm vùng nhận diện mặt.
    - Mỗi frame push 1 giá trị RMS vào lịch sử.
    - Thanh mới nhất (phải) sáng trắng, thanh cũ mờ dần sang xám.
    - Đối xứng trên/dưới quanh đường trung tâm.
    - Nền overlay mờ để không cạnh tranh với video.
    """
    N_BARS     = 60    # số thanh hiển thị (~2 giây ở 30fps)
    BAR_W      = 3     # pixel rộng mỗi thanh
    BAR_GAP    = 2     # khoảng cách giữa thanh
    MAX_HALF_H = 18    # chiều cao nửa trên/dưới tối đa (px)
    MIN_HALF_H = 2     # chiều cao tối thiểu khi im lặng

    def __init__(self):
        self.history = np.zeros(self.N_BARS)   # RMS history [0–1]

    def push(self, volume: float):
        """Thêm giá trị RMS mới — gọi mỗi frame."""
        self.history = np.roll(self.history, -1)
        self.history[-1] = float(np.clip(volume, 0.0, 1.0))

    def draw(self, frame):
        h, w = frame.shape[:2]

        total_w = self.N_BARS * (self.BAR_W + self.BAR_GAP) - self.BAR_GAP
        cx      = w // 2                      # center x của waveform
        x0      = cx - total_w // 2           # điểm bắt đầu
        cy      = h - 28                      # center y (dòng giữa)

        # ── Overlay nền mờ ────────────────────────────────────────────
        pad = 6
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (x0 - pad, cy - self.MAX_HALF_H - pad),
                      (x0 + total_w + pad, cy + self.MAX_HALF_H + pad),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # ── Vẽ từng thanh ─────────────────────────────────────────────
        for i, amp in enumerate(self.history):
            bx = x0 + i * (self.BAR_W + self.BAR_GAP)

            half_h = int(self.MIN_HALF_H + amp * (self.MAX_HALF_H - self.MIN_HALF_H))

            # Thanh mới (bên phải) sáng trắng; thanh cũ mờ dần
            t      = i / (self.N_BARS - 1)         # 0 = cũ nhất, 1 = mới nhất
            bright = int(60 + t * 195)              # 60 → 255
            color  = (bright, bright, bright)

            cv2.rectangle(frame,
                          (bx, cy - half_h),
                          (bx + self.BAR_W - 1, cy + half_h),
                          color, -1)

        # ── Đường trung tâm mỏng ──────────────────────────────────────
        cv2.line(frame, (x0, cy), (x0 + total_w, cy), (60, 60, 60), 1)


def draw_stream_status(frame, face_active, audio_active, fusion_mode):
    """Vẽ status indicators."""
    h, w = frame.shape[:2]
    y = h - 35

    statuses = [
        (f"FACE {'ON' if face_active else 'OFF'}",
         (0, 200, 0) if face_active else (100, 100, 100)),
        (f"AUDIO {'ON' if audio_active else 'OFF'}",
         (0, 200, 200) if audio_active else (100, 100, 100)),
        (f"FUSION {'ON' if fusion_mode else 'OFF'}",
         (200, 200, 0) if fusion_mode else (100, 100, 100)),
    ]
    x = 10
    for text, color in statuses:
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        x += 130


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    # Kiểm tra files
    for path, name in [(args.detector,   'Face Detector'),
                       (args.face_model, 'Face Emotion Model')]:
        if not os.path.exists(path):
            print(f"[ERROR] {name} not found: {path}")
            return

    os.makedirs('results', exist_ok=True)

    print("\n" + "=" * 55)
    print("  Multimodal Emotion Recognition — Demo")
    print("=" * 55)
    print(f"  Face detector : {os.path.basename(args.detector)}")
    print(f"  Face model    : {args.face_model}")
    print(f"  Audio model   : {args.audio_model}")
    print(f"  Fusion weights: {args.fusion}")
    print("=" * 55)

    # ── Load Face Pipeline ─────────────────────
    face_pipeline = FaceEmotionPipeline(
        detector_path      = args.detector,
        emotion_model_path = args.face_model,
        class_indices_path = args.classes,
        det_conf_threshold = args.det_thresh,
    )

    # ── Load Audio Predictor ───────────────────
    audio_buffer = None
    audio_active = False

    if not args.no_audio and os.path.exists(args.audio_model):
        try:
            audio_predictor = AudioPredictor(args.audio_model, args.classes)
            audio_buffer    = RealtimeAudioBuffer(audio_predictor)
            audio_buffer.start()
            audio_active = True
            print("[AUDIO] Stream started")
        except Exception as e:
            print(f"[AUDIO] Failed to start: {e}")
            print("[AUDIO] Install: pip install sounddevice librosa")
    else:
        print("[AUDIO] Skipped (no model or --no-audio)")

    # ── Load Fusion Weights ────────────────────
    fusion = LateFusion(w_face=0.5, w_audio=0.5, mode='confidence')
    if os.path.exists(args.fusion):
        fusion.load(args.fusion)

    # ── Webcam ─────────────────────────────────
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.cam}")
        return

    print("\n  q: Quit | s: Screenshot | f: Toggle fusion | k: Keypoints | b: Bars")
    print("  h: Ẩn/hiện toàn bộ UI (chỉ giữ bbox + nhãn kết quả)")
    print("  m: Toggle fusion mode (confidence / fixed)")
    print("  +/-: Detection threshold\n")

    show_kps    = True
    show_bars   = True
    show_labels = True   # h: ẩn/hiện toàn bộ UI, chỉ giữ bbox + nhãn kết quả
    fusion_mode = True
    screenshot_count = 0
    frame_count = 0
    t_start = datetime.now()

    # Default audio result
    default_audio = ('neutral', 0.0, np.ones(7) / 7)
    waveform      = IPhoneWaveform()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # ── Face Stream ────────────────────────
        face_results = face_pipeline.process(frame)

        # ── Audio Stream ───────────────────────
        audio_result = audio_buffer.get_result() if audio_active else default_audio
        audio_volume = audio_buffer.get_volume()  if audio_active else 0.0

        # ── Lưu nhãn face gốc trước khi fusion ghi đè ─────────────────
        for r in face_results:
            r['face_emotion'] = r['emotion']
            r['face_conf']    = r['confidence']

        # ── Fusion ─────────────────────────────
        if fusion_mode and face_results and audio_active:
            audio_probs = audio_result[2]
            for r in face_results:
                fused_em, fused_conf, fused_probs = fusion.fuse(
                    r['probs'], audio_probs)
                r['emotion']    = fused_em
                r['confidence'] = fused_conf
                r['probs']      = fused_probs

        # ── Waveform (luôn cập nhật history, chỉ vẽ khi show_labels) ───
        waveform.push(audio_volume)
        if show_labels:
            waveform.draw(frame)
            if audio_active:
                draw_audio_label(frame, audio_result)

        frame_out = face_pipeline.draw(frame, face_results,
                                       show_kps=show_kps and show_labels,
                                       show_bars=show_bars and show_labels)

        # ── Nhãn FACE gốc (dưới bbox, chỉ khi fusion ON + show_labels) ─
        if show_labels and fusion_mode and audio_active:
            for r in face_results:
                x1, y1, x2, y2 = r['bbox']
                face_em   = r.get('face_emotion', '')
                face_conf = r.get('face_conf', 0.0)
                color     = EMOTION_COLORS.get(face_em, (160, 160, 160))
                label     = f"face: {face_em} {face_conf*100:.0f}%"
                cv2.putText(frame_out, label,
                            (x1, y2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        # ── FPS & Status ───────────────────────
        frame_count += 1
        elapsed = (datetime.now() - t_start).total_seconds()
        fps = frame_count / elapsed if elapsed > 0 else 0

        if show_labels:
            cv2.putText(frame_out, f"FPS:{fps:.1f}",
                        (10, frame_out.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            draw_stream_status(frame_out, bool(face_results),
                               audio_active, fusion_mode)

        cv2.imshow('Multimodal Emotion Recognition', frame_out)

        # ── Keys ───────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_count += 1
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = f'results/screenshot_{ts}.png'
            cv2.imwrite(path, frame_out)
            print(f"Screenshot: {path}")
        elif key == ord('f'):
            fusion_mode = not fusion_mode
            print(f"Fusion: {'ON' if fusion_mode else 'OFF'}")
        elif key == ord('h'):
            show_labels = not show_labels
            print(f"Labels: {'ON' if show_labels else 'OFF (clean view)'}")
        elif key == ord('k'):
            show_kps = not show_kps
        elif key == ord('b'):
            show_bars = not show_bars
        elif key in (ord('+'), ord('=')):
            face_pipeline.detector.conf_thresh = min(
                0.95, face_pipeline.detector.conf_thresh + 0.05)
        elif key == ord('-'):
            face_pipeline.detector.conf_thresh = max(
                0.10, face_pipeline.detector.conf_thresh - 0.05)
        elif key == ord('m'):
            fusion.mode = 'fixed' if fusion.mode == 'confidence' else 'confidence'
            print(f"Fusion mode: {fusion.mode.upper()}")

    # Cleanup
    if audio_buffer:
        audio_buffer.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Demo closed.")


if __name__ == '__main__':
    main()
