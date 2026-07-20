import cv2
import numpy as np

GREEN_RATIO_MIN  = 0.08
TEXTURE_VAR_MIN  = 40.0
EDGE_DENSITY_MIN = 0.03
EDGE_DENSITY_MAX = 0.55
SATURATION_MIN   = 25.0

_face_cascade  = None
_body_cascade  = None
_upper_cascade = None

def _load_cascades():
    global _face_cascade, _body_cascade, _upper_cascade
    if _face_cascade is None:
        _face_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        _body_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
        _upper_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")


class LeafDetector:
    """
    Returns label: 'leaf' | 'human' | 'other'
    Dict keys match what SupervisorAgent expects:
      label, is_leaf, is_human, confidence, green_ratio, skin_ratio, reason
    """

    def check(self, img_bgr, img_hsv, img_gray) -> dict:
        _load_cascades()
        
        if img_bgr is None or img_hsv is None or img_gray is None:
            return self._result("other", 0.0, 0.0, 0.0, "Missing image arrays")

        total = img_bgr.shape[0] * img_bgr.shape[1]

        # --- Human detection ---
        small  = cv2.resize(img_gray, (320, 240))
        faces  = _face_cascade.detectMultiScale(small, 1.1, 4, minSize=(20, 20))
        # Only run body/upper if no face found (saves time)
        if len(faces) > 0:
            bodies, upper = [], []
        else:
            bodies = _body_cascade.detectMultiScale(small, 1.1, 3, minSize=(30, 60))
            upper  = _upper_cascade.detectMultiScale(small, 1.1, 3, minSize=(30, 40))

        skin_mask  = cv2.inRange(img_hsv, np.array([0, 20, 70]), np.array([25, 180, 255]))
        skin_ratio = cv2.countNonZero(skin_mask) / total

        human_detected = len(faces) > 0 or len(bodies) > 0 or len(upper) > 0

        if human_detected or skin_ratio > 0.35:
            conf = 0.90 if human_detected else round(min(skin_ratio * 2.5, 0.85), 2)
            return self._result("human", conf, 0.0, skin_ratio, "Human detected in frame")

        # --- Vegetation signals ---
        green_mask = cv2.inRange(img_hsv, np.array([18, 25, 30]), np.array([100, 255, 255]))
        green_ratio  = cv2.countNonZero(green_mask) / total
        texture_var  = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        edges        = cv2.Canny(img_gray, 40, 120)
        edge_density = cv2.countNonZero(edges) / total
        green_sat    = float(np.mean(img_hsv[:, :, 1][green_mask > 0])) if cv2.countNonZero(green_mask) > 0 else 0.0

        sky_ratio  = cv2.countNonZero(cv2.inRange(img_hsv, np.array([90, 0, 160]),  np.array([140, 80, 255])))  / total
        soil_ratio = cv2.countNonZero(cv2.inRange(img_hsv, np.array([5,  10, 20]),  np.array([25, 120, 140])))  / total

        score   = 0.0
        reasons = []

        if green_ratio >= GREEN_RATIO_MIN:
            score += 0.40
        else:
            reasons.append(f"low green ({green_ratio*100:.1f}%)")

        if texture_var >= TEXTURE_VAR_MIN:
            score += 0.20
        else:
            reasons.append(f"low texture ({texture_var:.1f})")

        if EDGE_DENSITY_MIN <= edge_density <= EDGE_DENSITY_MAX:
            score += 0.20
        else:
            reasons.append(f"edge density {edge_density:.3f}")

        if green_sat >= SATURATION_MIN:
            score += 0.10
        else:
            reasons.append(f"low sat ({green_sat:.1f})")

        if sky_ratio > 0.35:
            score -= 0.30
            reasons.append("sky dominant")
        if soil_ratio > 0.40:
            score -= 0.20
            reasons.append("soil dominant")

        score = float(np.clip(score, 0.0, 1.0))

        if score >= 0.50:
            return self._result("leaf", round(score, 3), round(green_ratio, 3), skin_ratio, "Leaf / vegetation detected")

        reason = "Non-vegetation: " + (", ".join(reasons) if reasons else "low score")
        return self._result("other", round(score, 3), round(green_ratio, 3), skin_ratio, reason)

    def _result(self, label: str, conf: float, green_ratio: float, skin_ratio: float, reason: str) -> dict:
        return {
            "label":       label,
            "is_leaf":     label == "leaf",
            "is_human":    label == "human",
            "confidence":  conf,
            "green_ratio": green_ratio,
            "skin_ratio":  skin_ratio,
            "reason":      reason,
        }

    # _decode removed as it is now redundant
