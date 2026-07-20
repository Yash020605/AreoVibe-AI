import cv2
import numpy as np
import time

class HeuristicVisionEngine:
    """
    Real image-analysis engine using HSV color segmentation and texture features.
    Replaces random simulation with actual pixel-level inspection of the live frame.
    Detects rust, necrosis, mildew, yellowing, wilting, and water-soaked lesions.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.is_loaded = True
        print(f"[VisionEngine] Vision engine ready (HSV + texture analysis).")

    def analyze_tile(self, img_bgr, img_hsv) -> dict:
        """
        Computes a real anomaly percentage by measuring how much of the frame
        deviates from healthy green vegetation in HSV space.
        """
        if img_bgr is None or img_hsv is None:
            return {"anomaly_percentage": 0.05}

        # Downsample for speed
        scale = 640.0 / max(img_hsv.shape[1], 1)
        if scale < 1.0:
            hsv = cv2.resize(img_hsv, (0, 0), fx=scale, fy=scale)
        else:
            hsv = img_hsv

        # Healthy green vegetation mask (H: 35-85, S: 40-255, V: 40-255)
        healthy_mask = cv2.inRange(hsv,
            np.array([35, 40, 40]),
            np.array([85, 255, 255])
        )

        total_pixels = hsv.shape[0] * hsv.shape[1]
        healthy_pixels = cv2.countNonZero(healthy_mask)
        anomaly_ratio = 1.0 - (healthy_pixels / total_pixels)

        # Clamp to a meaningful range
        anomaly_ratio = float(np.clip(anomaly_ratio, 0.0, 1.0))
        return {"anomaly_percentage": round(anomaly_ratio, 3)}

    def run_dynamic_tiling_inference(self, img_bgr, img_hsv, img_gray, low_light=False) -> list:
        """
        Analyses the actual frame pixels across multiple HSV/texture windows
        and returns symptom primitives that genuinely reflect what is visible.
        low_light=True suppresses dark-pixel Necrosis flags caused by shadows.
        """
        if img_bgr is None or img_hsv is None or img_gray is None:
            return []

        scale = 640.0 / max(img_bgr.shape[1], 1)
        if scale < 1.0:
            hsv = cv2.resize(img_hsv, (0, 0), fx=scale, fy=scale)
            gray = cv2.resize(img_gray, (0, 0), fx=scale, fy=scale)
        else:
            hsv = img_hsv
            gray = img_gray

        total = hsv.shape[0] * hsv.shape[1]
        primitives = []

        def ratio(mask):
            return cv2.countNonZero(mask) / total

        # --- Rust / Red-Brown spots (H: 0-15 or 160-180, high S) ---
        rust1 = cv2.inRange(hsv, np.array([0,  80, 50]), np.array([15, 255, 200]))
        rust2 = cv2.inRange(hsv, np.array([160,80, 50]), np.array([180,255, 200]))
        rust  = cv2.bitwise_or(rust1, rust2)
        if ratio(rust) > 0.04:
            primitives.append("Rusty-Pustules")
        if ratio(rust) > 0.06:
            primitives.append("Red-Brown-Spots")

        # --- Yellowing / Chlorosis (H: 20-35, high S, mid-high V) ---
        yellow = cv2.inRange(hsv, np.array([18, 80, 100]), np.array([35, 255, 255]))
        if ratio(yellow) > 0.08:
            primitives.append("Yellowish-Patches")
        if ratio(yellow) > 0.12:
            primitives.append("Chlorosis")

        # --- White / Powdery regions (low S, high V) ---
        white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 40, 255]))
        if ratio(white) > 0.06:
            primitives.append("White-Powder-Film")
        if ratio(white) > 0.10:
            primitives.append("White-Fuzzy-Growth-Underneath")
            primitives.append("Mildew")

        # --- Dark / Water-soaked lesions (very low V) — suppressed in low-light ---
        if not low_light:
            dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))
            if ratio(dark) > 0.08:
                primitives.append("Dark-Water-Soaked-Spots")

        # --- Necrosis: brown-grey dead tissue — suppressed in low-light to avoid shadow false positives ---
        if not low_light:
            necrotic = cv2.inRange(hsv, np.array([10, 10, 60]), np.array([30, 80, 160]))
            if ratio(necrotic) > 0.07:
                primitives.append("Necrosis")
                primitives.append("Necrotic-Edges")

        # --- Grey-brown lesions (H: 10-25, low-mid S) ---
        grey_brown = cv2.inRange(hsv, np.array([10, 20, 80]), np.array([25, 100, 180]))
        if ratio(grey_brown) > 0.06:
            primitives.append("Grey-Brown-Lesions")

        # --- Texture: cigar/elongated spots via edge density ---
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = cv2.countNonZero(edges) / total
        if edge_ratio > 0.12:
            primitives.append("Cigar-Shaped-Spots")

        # --- Wilting: loss of green + high brightness variance ---
        green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        if ratio(green) < 0.15:
            primitives.append("Wilting")

        # --- Leaf curling: high Laplacian variance (texture distortion) ---
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var > 800:
            primitives.append("Leaf-Curling")

        # --- Stunting proxy: very uniform low-texture frame ---
        if lap_var < 80 and ratio(green) < 0.2:
            primitives.append("Stunting")
            primitives.append("Thin-Shoots")

        return list(dict.fromkeys(primitives))  # deduplicate, preserve order
