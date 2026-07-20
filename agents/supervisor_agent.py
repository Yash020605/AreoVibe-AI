import random
import cv2
import numpy as np
from state import AgentState
from utils.crop_detector import CropDetector
from utils.leaf_detector import LeafDetector

LOW_LIGHT_THRESHOLD = 60

class SupervisorAgent:
    def __init__(self):
        self.crop_detector = CropDetector()
        self.leaf_detector = LeafDetector()

    def __call__(self, state: AgentState) -> AgentState:
        if "log_stream" not in state:
            state["log_stream"] = []

        def log(msg):
            try:
                print(msg)
            except UnicodeEncodeError:
                print(msg.encode("ascii", "replace").decode("ascii"))
            state["log_stream"].append(msg)

        log("[Supervisor] Initializing Context Preservation...")
        disp_path = state['image_path'] if isinstance(state['image_path'], str) else 'ImageBytes'
        log(f"[Supervisor] Tracking Image: {disp_path} | GPS: {state['gps_coordinates']}")

        image_input = state['image_path']
        if isinstance(image_input, (bytes, bytearray)):
            np_arr = np.frombuffer(image_input, np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            bgr = cv2.imread(image_input)

        if bgr is None:
            log("[Supervisor] Error: Could not decode image!")
            state["pipeline_status"] = "Failed"
            return state

        # Create pre-computed arrays
        state["image_bgr"] = bgr
        state["image_hsv"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        state["image_gray"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Check leaf presence
        leaf_result = self.leaf_detector.check(state["image_bgr"], state["image_hsv"], state["image_gray"])
        state["leaf_detected"] = leaf_result["is_leaf"]
        
        if not state["leaf_detected"]:
            log(f"[Supervisor] WARNING: No leaf detected. (conf: {leaf_result['confidence']:.2f})")
            state["pipeline_status"] = "Failed"
            return state
            
        log(f"[Supervisor] ✅ Leaf detected (conf: {leaf_result['confidence']:.2f}, green: {leaf_result['green_ratio']*100:.1f}%)")

        # Lighting check
        avg_brightness = np.mean(state["image_gray"])
        if avg_brightness < LOW_LIGHT_THRESHOLD:
            log(f"[Supervisor] WARNING: Low light detected! (brightness={avg_brightness:.1f})")
            state["low_light"] = True
        else:
            log(f"[Supervisor] Lighting OK: avg brightness={avg_brightness:.1f}")
            state["low_light"] = False

        if "rescan_count" not in state:
            state["rescan_count"] = 0
            state["dense_regions_found"] = 0

        state["pipeline_status"] = "Processing"
        return state
