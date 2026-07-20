from utils.heuristic_vision_engine import HeuristicVisionEngine
from state import AgentState

class VisionAgent:
    def __init__(self):
        self.engine = HeuristicVisionEngine()

    def __call__(self, state: AgentState) -> AgentState:
        if "log_stream" not in state:
            state["log_stream"] = []

        def log(msg):
            try:
                print(msg)
            except UnicodeEncodeError:
                print(msg.encode("ascii", "replace").decode("ascii"))
            state["log_stream"].append(msg)

        disp_path = state['image_path'] if isinstance(state['image_path'], str) else 'ImageBytes'
        log(f"[Vision Agent] Processing image: {disp_path}")

        low_light = state.get("low_light", False)
        if low_light:
            log("[Vision Agent] Low-light mode active — shadow Necrosis suppression ON.")

        if state.get("needs_deconvolution", False):
            log("[Vision Agent] Applying internal ONNX AI-Deconvolution prior to scan...")

        is_rescan = state.get("visual_rescan_requested", False)

        if is_rescan:
            log("[Vision Agent] Dynamic Rescan Requested! Executing deep dive.")
            primitives = self.engine.run_dynamic_tiling_inference(state["image_bgr"], state["image_hsv"], state["image_gray"], low_light=low_light)
            state["symptom_primitives"] = primitives
            state["dense_regions_found"] += 1
            state["visual_rescan_requested"] = False
            log(f"[Vision Agent] Extracted Symptom Primitives: {primitives}")
            return state

        result = self.engine.analyze_tile(state["image_bgr"], state["image_hsv"])
        log(f"[Vision Agent] Initial scan detected anomaly: {result['anomaly_percentage'] * 100:.1f}%")

        if result["anomaly_percentage"] > 0.05:
            log("[Vision Agent] Anomaly > 5%. Triggering Dynamic Tiling...")
            primitives = self.engine.run_dynamic_tiling_inference(state["image_bgr"], state["image_hsv"], state["image_gray"], low_light=low_light)
            state["symptom_primitives"] = primitives
            state["dense_regions_found"] += 1
            log(f"[Vision Agent] Extracted Symptom Primitives: {primitives}")
        else:
            log("[Vision Agent] No significant anomalies found.")
            state["symptom_primitives"] = []

        return state
