from utils.rag_engine import PlantPathologyRAG
from state import AgentState

class AnalysisAgent:
    def __init__(self):
        self.rag = PlantPathologyRAG()

    def __call__(self, state: AgentState) -> AgentState:
        print(f"[Analysis Agent] Using Plant Pathology KB for crop: '{state['crop_type']}'")
        
        primitives = state.get("symptom_primitives", [])
        if not primitives:
            print("[Analysis Agent] No symptom primitives provided. Marking as Healthy.")
            state["diagnoses"] = [{"diagnosis": "Healthy", "confidence": 1.0}]
            state["confidence_score"] = 1.0
            return state

        result = self.rag.retrieve_pathology(state['crop_type'], primitives)
        
        confidence = result['confidence']
        state["diagnoses"] = [result]
        state["confidence_score"] = confidence
        
        print(f"[Analysis Agent] Diagnosis: {result['diagnosis'].get('disease', 'Unknown')} | Confidence: {confidence}")
        
        # The core A2A requirement
        if confidence < 0.6 and state.get("rescan_count", 0) < 2:
            print("[Analysis Agent] Confidence < 0.6. Triggering A2A Visual Re-scan request to Vision Agent.")
            state["visual_rescan_requested"] = True
            state["rescan_count"] = state.get("rescan_count", 0) + 1
        else:
            if confidence < 0.6:
                print("[Analysis Agent] Confidence still low, but max rescans reached. Proceeding with caveat.")
            
        return state
