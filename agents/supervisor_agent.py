import random
from state import AgentState

class SupervisorAgent:
    def __init__(self):
        pass

    def __call__(self, state: AgentState) -> AgentState:
        print("[Supervisor] Initializing Context Preservation...")
        print(f"[Supervisor] Tracking Image: {state['image_path']} | GPS: {state['gps_coordinates']}")
        
        # Initialize internal tracking if missing
        if "rescan_count" not in state:
            state["rescan_count"] = 0
            state["dense_regions_found"] = 0
        
        # Check image quality constraints
        is_blurry = state.get("is_blurry", None)
        if is_blurry is None:
            # Simulate a 10% chance the input image is blurry
            is_blurry = random.random() < 0.1
            state["is_blurry"] = is_blurry
            
        if state["is_blurry"]:
            print("[Supervisor] WARNING: Input image flagged as Blurry!")
            # 50/50 decide whether to AI-Deconvolve or flag unreliable
            if random.random() < 0.5:
                print("[Supervisor] Decision: Attempting AI-Deconvolution on image array.")
                state["needs_deconvolution"] = True
                state["pipeline_status"] = "Processing"
            else:
                print("[Supervisor] Decision: Image too degraded. Flagging data as 'Unreliable' for the final report.")
                state["pipeline_status"] = "Unreliable"
                return state
        else:
            state["pipeline_status"] = "Processing"
            
        return state
