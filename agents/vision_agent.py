from utils.onnx_engine import PlantPathologyONNXEngine
from state import AgentState

class VisionAgent:
    def __init__(self):
        self.engine = PlantPathologyONNXEngine("models/yolo_plant_v8.onnx")

    def __call__(self, state: AgentState) -> AgentState:
        print("[Vision Agent] Processing image:", state["image_path"])
        
        # If the image was flagged as blurry and deconvolved by Supervisor, we note it.
        if state.get("needs_deconvolution", False):
            print("[Vision Agent] Applying internal ONNX AI-Deconvolution prior to scan...")
            
        is_rescan = state.get("visual_rescan_requested", False)
        
        if is_rescan:
            print("[Vision Agent] Dynamic Rescan Requested! Executing 2x Resolution deep dive.")
            primitives = self.engine.run_dynamic_tiling_inference(state["image_path"])
            state["symptom_primitives"] = primitives
            state["dense_regions_found"] += 1
            state["visual_rescan_requested"] = False # Reset for the loop
            print(f"[Vision Agent] Extracted Symptom Primitives: {primitives}")
            return state

        # Initial Standard Scan
        result = self.engine.analyze_tile(state["image_path"])
        print(f"[Vision Agent] Initial scan detected anomaly: {result['anomaly_percentage'] * 100:.1f}%")
        
        if result["anomaly_percentage"] > 0.05:
            print("[Vision Agent] Anomaly > 5%. Triggering Dynamic Tiling...")
            primitives = self.engine.run_dynamic_tiling_inference(state["image_path"])
            state["symptom_primitives"] = primitives
            state["dense_regions_found"] += 1
            print(f"[Vision Agent] Extracted Symptom Primitives: {primitives}")
        else:
            print("[Vision Agent] No significant anomalies found.")
            state["symptom_primitives"] = []
            
        return state
