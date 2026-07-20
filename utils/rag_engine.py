import json
import os

class PlantPathologyRAG:
    def __init__(self, db_path='knowledge_base/pathologies.json'):
        self.db_path = db_path
        self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r', encoding='utf-8') as f:
                self.kb = json.load(f)
        else:
            self.kb = {}
            
        self.global_kb = {}
        for entries in self.kb.values():
            self.global_kb.update(entries)

    def retrieve_pathology(self, crop_type: str, symptoms: list) -> dict:
        crop_type = (crop_type or "").lower().strip()
        symptoms_set = set(symptoms)

        # Build search space: crop-specific first, then global fallback
        search_space = self.kb.get(crop_type, {})
        if not search_space:
            search_space = self.global_kb

        best_match = None
        highest_iou = 0.0

        for disease, details in search_space.items():
            kb_symptoms = set(details.get("primitives", []))
            if not kb_symptoms:
                continue
            intersection = len(symptoms_set & kb_symptoms)
            union = len(symptoms_set | kb_symptoms)
            iou = intersection / union if union > 0 else 0.0
            if iou > highest_iou:
                highest_iou = iou
                # Build a clean match dict — never overwrite with .update()
                best_match = {
                    "disease_name": disease,
                    "diagnosis_en": details.get("diagnosis_en", disease),
                    "diagnosis_hi": details.get("diagnosis_hi", ""),
                    "diagnosis_mr": details.get("diagnosis_mr", ""),
                    "treatment_en": details.get("treatment_en", "Consult local agronomist."),
                    "treatment_hi": details.get("treatment_hi", ""),
                    "treatment_mr": details.get("treatment_mr", ""),
                    "primitives":   details.get("primitives", []),
                }

        if best_match is None or highest_iou == 0.0:
            return {
                "diagnosis": {
                    "disease_name": "Generic Stress",
                    "diagnosis_en": "Unidentified Stress / Possible Deficiency",
                    "diagnosis_hi": "अज्ञात तनाव / संभावित कमी",
                    "diagnosis_mr": "अज्ञात ताण / संभाव्य कमतरता",
                    "treatment_en": "Apply organic micronutrients. Ensure proper watering and soil health.",
                    "treatment_hi": "जैविक सूक्ष्म पोषक तत्वों का प्रयोग करें।",
                    "treatment_mr": "सेंद्रिय सूक्ष्म पोषक तत्वांचा वापर करा.",
                },
                "confidence": 0.2
            }

        confidence = round(min(highest_iou * 0.9 + 0.1, 0.99), 2)
        return {"diagnosis": best_match, "confidence": confidence}
