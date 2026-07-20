import cv2
import numpy as np
import json
import os

class CropDetector:
    """
    HSV-based crop auto-detection using a profile database of 100+ crops.
    Scores each crop profile against the dominant color/texture of the frame.
    """

    def __init__(self, db_path='knowledge_base/crop_profiles.json'):
        with open(db_path, 'r', encoding='utf-8') as f:
            self.profiles = json.load(f)

    def detect(self, img_bgr, img_hsv, img_gray=None, top_n=3) -> list:
        """
        Analyse the frame and return a ranked list of (crop_name, confidence) tuples.
        """
        if img_bgr is None or img_hsv is None:
            return []

        if img_gray is None:
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
        total = img_bgr.shape[0] * img_bgr.shape[1]

        # Dominant hue/sat/val (median of vegetation pixels only)
        veg_mask = cv2.inRange(img_hsv,
            np.array([20, 20, 30]),
            np.array([100, 255, 255])
        )
        veg_pixels = cv2.countNonZero(veg_mask)
        veg_ratio  = veg_pixels / total

        if veg_ratio < 0.05:
            # Mostly non-vegetation frame — can't reliably detect
            return [("unknown", 0.0)]

        veg_idx = veg_mask > 0
        dom_h = float(np.median(img_hsv[:, :, 0][veg_idx].astype(float)))
        dom_s = float(np.median(img_hsv[:, :, 1][veg_idx].astype(float)))
        dom_v = float(np.median(img_hsv[:, :, 2][veg_idx].astype(float)))

        # Texture: Laplacian variance
        lap_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()

        scores = {}
        for crop, p in self.profiles.items():
            score = 0.0

            # Hue match (most discriminative)
            h_range = p["hue_max"] - p["hue_min"]
            h_center = (p["hue_min"] + p["hue_max"]) / 2
            h_dist = abs(dom_h - h_center)
            h_score = max(0.0, 1.0 - (h_dist / (h_range + 1e-5)))
            score += h_score * 0.5

            # Saturation match
            s_center = (p["sat_min"] + p["sat_max"]) / 2
            s_range  = p["sat_max"] - p["sat_min"]
            s_dist   = abs(dom_s - s_center)
            s_score  = max(0.0, 1.0 - (s_dist / (s_range + 1e-5)))
            score += s_score * 0.25

            # Value/brightness match
            v_center = (p["val_min"] + p["val_max"]) / 2
            v_range  = p["val_max"] - p["val_min"]
            v_dist   = abs(dom_v - v_center)
            v_score  = max(0.0, 1.0 - (v_dist / (v_range + 1e-5)))
            score += v_score * 0.15

            # Texture match
            tex = p.get("texture", "medium")
            tex_score = self._texture_score(lap_var, tex)
            score += tex_score * 0.10

            scores[crop] = round(score, 4)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Normalize top scores to confidence %
        top_val = ranked[0][1] if ranked else 1.0
        result = [(crop, round(s / top_val, 3)) for crop, s in ranked]
        return result

    def _texture_score(self, lap_var: float, texture_label: str) -> float:
        """Map Laplacian variance to a texture category score."""
        # Approximate lap_var ranges per texture type
        ranges = {
            "fine":       (0,    200),
            "medium":     (100,  600),
            "coarse":     (400, 1500),
            "very_coarse":(800, 3000),
            "glossy":     (0,    300),
        }
        lo, hi = ranges.get(texture_label, (0, 1000))
        center = (lo + hi) / 2
        span   = (hi - lo) / 2 + 1e-5
        dist   = abs(lap_var - center)
        return max(0.0, 1.0 - dist / span)

    # _decode removed as it is now redundant
