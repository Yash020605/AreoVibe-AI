from utils.rag_engine import PlantPathologyRAG
from state import AgentState

# Confidence threshold below which crop is considered wild/non-commercial
WILD_CROP_THRESHOLD = 0.4


class AnalysisAgent:
    def __init__(self):
        self.rag = PlantPathologyRAG()

    def __call__(self, state: AgentState) -> AgentState:
        if "log_stream" not in state:
            state["log_stream"] = []

        def log(msg):
            try:
                print(msg)
            except UnicodeEncodeError:
                print(msg.encode("ascii", "replace").decode("ascii"))
            state["log_stream"].append(msg)

        crop = state.get("crop_type", "unknown")
        crop_conf = state.get("crop_detection_confidence", 1.0)  # 1.0 = manual selection (trusted)

        # --- Task 2: Wild/Out-of-Distribution vegetation check ---
        if state.get("auto_detect_crop", False) and crop_conf < WILD_CROP_THRESHOLD:
            log(f"[Analysis Agent] Crop detection confidence {crop_conf:.2f} < {WILD_CROP_THRESHOLD}. "
                f"Flagging as Wild/Non-Commercial Vegetation.")
            wild_diag = self.rag.kb.get("wild_vegetation", {})
            state["diagnoses"] = [{"diagnosis": wild_diag, "confidence": crop_conf}]
            state["confidence_score"] = crop_conf
            state["wild_vegetation"] = True
            return state

        state["wild_vegetation"] = False
        log(f"[Analysis Agent] Using Plant Pathology KB for crop: '{crop}'")

        primitives = state.get("symptom_primitives", [])
        if not primitives:
            log("[Analysis Agent] No symptom primitives provided. Marking as Healthy.")
            state["diagnoses"] = [{"diagnosis": "Healthy", "confidence": 1.0}]
            state["confidence_score"] = 1.0
            return state

        result = self.rag.retrieve_pathology(crop, primitives)
        confidence = result['confidence']
        state["diagnoses"] = [result]
        state["confidence_score"] = confidence

        disease_name = (
            result['diagnosis'].get('disease_name', result['diagnosis'].get('disease', 'Unknown'))
            if isinstance(result['diagnosis'], dict) else result['diagnosis']
        )

        if len(primitives) > 1:
            log(f"[Analysis Agent] Multiple primitives detected.")
            log(f"[Analysis Agent] Primary Concern: {primitives[0]}")
            log(f"[Analysis Agent] Secondary Concern: {primitives[1]}")
            if isinstance(result['diagnosis'], dict):
                base_diag = result['diagnosis'].get('diagnosis_en', disease_name)
                result['diagnosis']['diagnosis_en'] = f"{base_diag} (Primary: {primitives[0]} | Secondary: {primitives[1]})"

        log(f"[Analysis Agent] Diagnosis: {disease_name} | Confidence: {confidence}")

        if confidence < 0.6 and state.get("rescan_count", 0) < 2:
            log("[Analysis Agent] Confidence < 0.6. Triggering A2A Visual Re-scan.")
            state["visual_rescan_requested"] = True
            state["rescan_count"] = state.get("rescan_count", 0) + 1
        elif confidence < 0.6:
            log("[Analysis Agent] Confidence still low, max rescans reached. Proceeding.")

        return state
