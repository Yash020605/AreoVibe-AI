from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    # Context Preservation Parameters
    image_path: str
    gps_coordinates: str
    crop_type: str
    
    # Pre-computed Image Data
    image_bgr: Any
    image_hsv: Any
    image_gray: Any
    
    # Vision Agent Outputs
    is_blurry: bool
    symptom_primitives: List[str]
    needs_deconvolution: bool
    dense_regions_found: int
    
    # Analysis Agent Outputs
    diagnoses: List[Dict[str, Any]]
    confidence_score: float
    
    # Supervisor / Flow Control
    visual_rescan_requested: bool
    rescan_count: int
    pipeline_status: str
    
    # UI Callbacks
    log_stream: List[str]
