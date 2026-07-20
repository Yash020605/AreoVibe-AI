from state import AgentState
from agents.supervisor_agent import SupervisorAgent
from agents.vision_agent import VisionAgent
from agents.analysis_agent import AnalysisAgent
from agents.coordination_agent import CoordinationAgent
import argparse

class A2APipeline:
    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.vision = VisionAgent()
        self.analysis = AnalysisAgent()
        self.coordination = CoordinationAgent()

    def run_live(self, initial_state: AgentState):
        """Yields state at each agent transition for live Streamlit UI streaming."""
        state = initial_state
        state = self.supervisor(state)
        yield state
        
        if state.get("pipeline_status") in ("Unreliable", "NoVegetation"):
            return
            
        while True:
            state = self.vision(state)
            yield state
            
            state = self.analysis(state)
            yield state
            
            state = self.coordination(state)
            yield state
            
            if state.get("visual_rescan_requested", False):
                if "log_stream" in state:
                    state["log_stream"].append("[Pipeline] A2A Routing: Returning to Vision Agent for Re-scan")
                yield state
                continue
            else:
                break
                
        return

    def run(self, initial_state: AgentState):
        state = initial_state
        print("=" * 50)
        print("STARTING A2A MULTI-AGENT PIPELINE")
        print("=" * 50)
        
        state = self.supervisor(state)
        if state.get("pipeline_status") in ("Unreliable", "NoVegetation"):
            self._print_final(state)
            return state
            
        while True:
            state = self.vision(state)
            state = self.analysis(state)
            state = self.coordination(state)
            
            if state.get("visual_rescan_requested", False):
                print("-" * 50)
                print("[Pipeline] A2A Routing: Returning to Vision Agent for Re-scan")
                print("-" * 50)
                continue
            else:
                break
                
        self._print_final(state)
        return state

    def _print_final(self, state):
        print("=" * 50)
        print("PIPELINE COMPLETE")
        print("FINAL STATE:")
        print(f"Context (GPS): {state.get('gps_coordinates')}")
        print(f"Final Pipeline Status: {state.get('pipeline_status')}")
        print(f"Rescans Triggered: {state.get('rescan_count')}")
        
        if state.get("pipeline_status") == "Processing":
            print(f"Final Diagnoses: {state.get('diagnoses')}")
        print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Crop-Agnostic Setup")
    parser.add_argument("--image", type=str, default="test_crop.jpg", help="Path to raw image capture")
    parser.add_argument("--crop_type", type=str, default="corn", help="Target crop type for knowledge base")
    parser.add_argument("--gps", type=str, default="34.05,-118.24", help="Context preservation GPS tags")
    args = parser.parse_args()
    
    pipeline = A2APipeline()
    initial_state = {
        "image_path": args.image,
        "gps_coordinates": args.gps,
        "crop_type": args.crop_type,
        "rescan_count": 0,
        "dense_regions_found": 0,
        "visual_rescan_requested": False
    }
    
    pipeline.run(initial_state)
