import json
import os

class PlantPathologyRAG:
    """
    Mock RAG Engine for the Plant Pathology Knowledge Base
    Retrieves symptom-based diagnoses for a given crop type.
    """
    def __init__(self, db_path='knowledge_base/pathologies.json'):
        self.db_path = db_path
        self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                self.kb = json.load(f)
        else:
            self.kb = {}

    def retrieve_pathology(self, crop_type: str, symptoms: list) -> dict:
        """
        Simulate a vector search comparing 'Symptom Primitives' against known pathologies.
        Returns the closest match pathology and a simulated confidence score.
        """
        if not crop_type or crop_type not in self.kb:
            return {"diagnosis": "Unknown Crop", "confidence": 0.0}

        crop_pathways = self.kb[crop_type]
        best_match = None
        highest_overlap = 0

        # Simple intersection over union (IoU) mock for RAG similarity
        symptoms_set = set(symptoms)
        
        for disease, details in crop_pathways.items():
            kb_symptoms = set(details.get("primitives", []))
            if not kb_symptoms:
                continue
                
            intersection = len(symptoms_set.intersection(kb_symptoms))
            union = len(symptoms_set.union(kb_symptoms))
            iou = intersection / union if union > 0 else 0
            
            if iou > highest_overlap:
                highest_overlap = iou
                best_match = {
                    "disease": disease,
                    "description": details.get("description", ""),
                    "treatment": details.get("treatment", "")
                }
                
        # If no overlap, confidence is 0
        if best_match is None:
            return {"diagnosis": "No matching pathology found", "confidence": 0.1}
            
        # Simulate base confidence from vector overlap, add some randomness for reality
        confidence = highest_overlap * 0.9 + 0.1
        
        return {
            "diagnosis": best_match,
            "confidence": round(confidence, 2)
        }
