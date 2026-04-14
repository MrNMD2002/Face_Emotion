"""
Face Detection + Emotion Recognition Pipeline
=============================================
Tích hợp:
  - Face Detector : InsightFace SCRFD (det_10g_fp16_dynamic.onnx)
  - Emotion Model : CNN trained trên FER2013

Model info (det_10g_fp16_dynamic.onnx):
  Input  : [batch, 3, H, W] — float16, dynamic size
  Outputs: 9 outputs = 3 strides (8, 16, 32) × (scores, bboxes, keypoints)
    - score_{stride} : [batch_anchors, 1]
    - bbox_{stride}  : [batch_anchors, 4]   — (left, top, right, bottom) distances
    - kps_{stride}   : [batch_anchors, 10]  — 5 keypoints × (x, y)
  Normalize: (pixel - 127.5) / 128.0
"""

import cv2
import numpy as np
import onnxruntime as ort
import json
import os
from scipy.special import softmax

# ===================== CONFIG =====================
IMG_SIZE     = 48
EMOTIONS     = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
FACE_MARGIN  = 0.20   # padding quanh mặt (20%)

EMOTION_COLORS = {
    'angry'   : (0,   0,   255),
    'disgust' : (0,   128, 0  ),
    'fear'    : (128, 0,   128),
    'happy'   : (0,   220, 220),
    'neutral' : (200, 200, 200),
    'sad'     : (255, 100, 0  ),
    'surprise': (0,   165, 255),
}
# ==================================================


# ──────────────────────────────────────────────────
# SCRFD Face Detector
# ──────────────────────────────────────────────────
class SCRFDDetector:
    """
    Wrapper cho InsightFace SCRFD face detection model.

    SCRFD Output format (9 outputs, 3 strides: 8, 16, 32):
      outputs[0], [1], [2]  → scores  [N, 1]
      outputs[3], [4], [5]  → bboxes  [N, 4]  (ltrb distances)
      outputs[6], [7], [8]  → kps     [N, 10] (5 keypoints × xy)
    """
    STRIDES      = [8, 16, 32]
    NUM_ANCHORS  = 2   # anchors mỗi vị trí

    def __init__(self, model_path, input_size=(640, 640), conf_threshold=0.45):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session      = ort.InferenceSession(model_path, providers=providers)
        self.input_name   = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_size   = input_size          # (W, H)
        self.conf_thresh  = conf_threshold

        # Pre-generate anchor centers
        self._anchors = self._generate_anchors(input_size)

        print(f"SCRFDDetector loaded: {os.path.basename(model_path)}")
        print(f"  Input size   : {input_size}")
        print(f"  Conf thresh  : {conf_threshold}")
        print(f"  Providers    : {self.session.get_providers()}")

    # ── Anchor generation ──────────────────────────
    def _generate_anchors(self, input_size):
        iw, ih = input_size
        anchors = {}
        for stride in self.STRIDES:
            h = ih // stride
            w = iw // stride
            centers = np.stack(
                np.mgrid[:h, :w][::-1], axis=-1
            ).reshape(-1, 2).astype(np.float32) * stride
            # Repeat NUM_ANCHORS lần cho mỗi vị trí
            centers = np.tile(
                centers.reshape(-1, 1, 2), (1, self.NUM_ANCHORS, 1)
            ).reshape(-1, 2)
            anchors[stride] = centers
        return anchors

    # ── Preprocess ────────────────────────────────
    def preprocess(self, img_bgr):
        """Resize + pad + normalize → float16 NCHW tensor."""
        h, w = img_bgr.shape[:2]
        iw, ih = self.input_size

        scale  = min(iw / w, ih / h)
        nw, nh = int(w * scale), int(h * scale)

        img_resized = cv2.resize(img_bgr, (nw, nh)).astype(np.float32)

        # Letterbox padding
        img_pad = np.zeros((ih, iw, 3), dtype=np.float32)
        img_pad[:nh, :nw] = img_resized

        # Normalize: SCRFD dùng mean=127.5, std=128
        img_pad = (img_pad - 127.5) / 128.0

        # HWC → NCHW, float16 (model yêu cầu fp16)
        tensor = img_pad.transpose(2, 0, 1)[np.newaxis].astype(np.float16)
        return tensor, scale

    # ── Decode outputs ────────────────────────────
    def _decode(self, outputs, scale, orig_hw):
        """Parse 9 SCRFD outputs → list of face dicts."""
        oh, ow = orig_hw
        faces  = []

        for i, stride in enumerate(self.STRIDES):
            scores = outputs[i].flatten().astype(np.float32)          # [N]
            bboxes = outputs[i + 3].astype(np.float32)                # [N, 4]
            kps    = outputs[i + 6].reshape(-1, 5, 2).astype(np.float32)  # [N, 5, 2]

            centers = self._anchors[stride]   # [N, 2]

            valid = scores >= self.conf_thresh
            if not valid.any():
                continue

            scores  = scores[valid]
            bboxes  = bboxes[valid]
            kps     = kps[valid]
            centers = centers[valid]

            # Decode bboxes: ltrb distances × stride → absolute coords → scale back
            x1 = np.clip((centers[:, 0] - bboxes[:, 0] * stride) / scale, 0, ow)
            y1 = np.clip((centers[:, 1] - bboxes[:, 1] * stride) / scale, 0, oh)
            x2 = np.clip((centers[:, 0] + bboxes[:, 2] * stride) / scale, 0, ow)
            y2 = np.clip((centers[:, 1] + bboxes[:, 3] * stride) / scale, 0, oh)

            # Decode keypoints
            kps_abs = np.zeros_like(kps)
            kps_abs[:, :, 0] = np.clip(
                (centers[:, 0:1] + kps[:, :, 0] * stride) / scale, 0, ow)
            kps_abs[:, :, 1] = np.clip(
                (centers[:, 1:2] + kps[:, :, 1] * stride) / scale, 0, oh)

            for j in range(len(scores)):
                faces.append({
                    'bbox' : [int(x1[j]), int(y1[j]), int(x2[j]), int(y2[j])],
                    'score': float(scores[j]),
                    'kps'  : kps_abs[j].astype(int).tolist()
                })

        return self._nms(faces)

    # ── NMS ───────────────────────────────────────
    def _nms(self, faces, iou_thresh=0.4):
        if not faces:
            return []
        faces = sorted(faces, key=lambda x: x['score'], reverse=True)
        keep  = []
        while faces:
            best = faces.pop(0)
            keep.append(best)
            faces = [f for f in faces
                     if self._iou(best['bbox'], f['bbox']) < iou_thresh]
        return keep

    def _iou(self, a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        ua    = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / (ua + 1e-6)

    # ── Public API ────────────────────────────────
    def detect(self, img_bgr):
        """
        Detect faces trong ảnh BGR.
        Returns: list of dict {'bbox': [x1,y1,x2,y2], 'score': float, 'kps': [[x,y]×5]}
        """
        tensor, scale = self.preprocess(img_bgr)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        return self._decode(outputs, scale, img_bgr.shape[:2])


# ──────────────────────────────────────────────────
# Face Cropper
# ──────────────────────────────────────────────────
def crop_face(img_bgr, bbox, margin=FACE_MARGIN):
    """
    Crop và preprocess 1 khuôn mặt thành tensor cho emotion model.

    Args:
        img_bgr : ảnh gốc (BGR numpy)
        bbox    : [x1, y1, x2, y2]
        margin  : padding quanh bbox (0.2 = 20%)

    Returns:
        face_tensor : numpy (1, 48, 48, 1) float32 [0,1]
        face_roi    : ảnh grayscale cropped (48×48) để hiển thị
    """
    h, w   = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox

    # Thêm margin
    bw = x2 - x1
    bh = y2 - y1
    x1 = max(0, int(x1 - bw * margin))
    y1 = max(0, int(y1 - bh * margin))
    x2 = min(w, int(x2 + bw * margin))
    y2 = min(h, int(y2 + bh * margin))

    face_crop = img_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None, None

    # Grayscale → resize → normalize
    face_gray    = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
    face_roi     = face_resized.copy()

    face_norm   = face_resized.astype(np.float32) / 255.0
    face_tensor = face_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    return face_tensor, face_roi


# ──────────────────────────────────────────────────
# Emotion Predictor
# ──────────────────────────────────────────────────
class EmotionPredictor:
    """
    Load emotion model từ ONNX (export từ PyTorch train.py).
    Dùng onnxruntime — nhất quán với SCRFDDetector.
    """
    def __init__(self, model_path, class_indices_path=None):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session    = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        if class_indices_path and os.path.exists(class_indices_path):
            with open(class_indices_path) as f:
                class_idx = json.load(f)
            self.idx_to_class = {v: k for k, v in class_idx.items()}
        else:
            self.idx_to_class = {i: e for i, e in enumerate(EMOTIONS)}

        print(f"EmotionPredictor loaded: {os.path.basename(model_path)}")
        print(f"  Providers: {self.session.get_providers()}")

    def predict(self, face_tensor):
        """
        Dự đoán cảm xúc từ face tensor (1, 48, 48, 1) float32.
        Returns: (emotion_label, confidence, all_probs)
        """
        # Chuyển (1,48,48,1) HWC → (1,1,48,48) NCHW cho PyTorch ONNX
        inp = face_tensor.transpose(0, 3, 1, 2).astype(np.float32)
        # Normalize về [-1, 1] (PyTorch model dùng mean=0.5, std=0.5)
        inp = (inp - 0.5) / 0.5

        logits = self.session.run(None, {self.input_name: inp})[0][0]
        probs  = softmax(logits).astype(np.float32)

        pred_idx   = int(np.argmax(probs))
        emotion    = self.idx_to_class[pred_idx]
        confidence = float(probs[pred_idx])
        return emotion, confidence, probs


# ──────────────────────────────────────────────────
# Complete Pipeline
# ──────────────────────────────────────────────────
class FaceEmotionPipeline:
    """
    Pipeline hoàn chỉnh: Ảnh/Frame → Detect Face → Crop → Predict Emotion

    Usage:
        pipeline = FaceEmotionPipeline(detector_path, emotion_model_path)
        results  = pipeline.process(frame)
        frame_out = pipeline.draw(frame, results)
    """

    def __init__(self, detector_path, emotion_model_path,
                 class_indices_path=None,
                 det_input_size=(640, 640),
                 det_conf_threshold=0.45,
                 face_margin=FACE_MARGIN):

        self.detector  = SCRFDDetector(detector_path, det_input_size, det_conf_threshold)
        self.predictor = EmotionPredictor(emotion_model_path, class_indices_path)
        self.margin    = face_margin

    def process(self, img_bgr):
        """
        Xử lý 1 ảnh/frame.

        Returns: list of dict {
            'bbox'      : [x1, y1, x2, y2],
            'score'     : float,
            'kps'       : [[x,y]×5],
            'emotion'   : str,
            'confidence': float,
            'probs'     : np.array (7,),
            'face_roi'  : np.array (48,48) grayscale
        }
        """
        faces   = self.detector.detect(img_bgr)
        results = []

        for face in faces:
            face_tensor, face_roi = crop_face(img_bgr, face['bbox'], self.margin)
            if face_tensor is None:
                continue

            emotion, conf, probs = self.predictor.predict(face_tensor)

            results.append({
                **face,
                'emotion'   : emotion,
                'confidence': conf,
                'probs'     : probs,
                'face_roi'  : face_roi
            })

        return results

    def draw(self, frame, results, show_kps=True, show_bars=True):
        """Vẽ kết quả lên frame và trả về frame đã annotated."""
        frame_out = frame.copy()

        for r in results:
            x1, y1, x2, y2 = r['bbox']
            emotion  = r['emotion']
            conf     = r['confidence']
            det_conf = r['score']
            color    = EMOTION_COLORS.get(emotion, (0, 255, 0))

            # Bounding box
            cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 2)

            # Emotion label
            label = f"{emotion.upper()} {conf*100:.0f}%"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame_out, (x1, y1-lh-14), (x1+lw+8, y1), color, -1)
            cv2.putText(frame_out, label, (x1+4, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

            # Face det score
            cv2.putText(frame_out, f"det:{det_conf:.2f}",
                        (x1, y2+16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Keypoints (5 điểm: mắt trái, mắt phải, mũi, miệng trái, miệng phải)
            if show_kps and r.get('kps'):
                kps_colors = [(255,0,0),(0,0,255),(0,255,0),(255,255,0),(255,0,255)]
                for k, (kx, ky) in enumerate(r['kps']):
                    cv2.circle(frame_out, (kx, ky), 3, kps_colors[k], -1)

            # Confidence bars (bên phải bbox)
            if show_bars:
                bx = x2 + 10
                if bx + 130 > frame_out.shape[1]:
                    bx = max(5, x1 - 130)
                for i, (prob, em) in enumerate(zip(r['probs'], EMOTIONS)):
                    by = y1 + i * 19
                    if by + 14 > frame_out.shape[0]:
                        break
                    filled = int(prob * 100)
                    ec = EMOTION_COLORS.get(em, (150,150,150))
                    cv2.rectangle(frame_out, (bx, by), (bx+100, by+13), (40,40,40), -1)
                    cv2.rectangle(frame_out, (bx, by), (bx+filled, by+13), ec, -1)
                    cv2.putText(frame_out,
                                f"{em[:3]}:{prob*100:.0f}%",
                                (bx+103, by+11),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (220,220,220), 1)

        # Face count
        cv2.putText(frame_out, f"Faces: {len(results)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50,255,50), 2)

        return frame_out
