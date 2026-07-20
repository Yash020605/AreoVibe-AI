import json
from state import AgentState

class CoordinationAgent:
    def __init__(self):
        pass

    def __call__(self, state: AgentState) -> AgentState:
        if "log_stream" not in state:
            state["log_stream"] = []
            
        def log(msg):
            try:
                print(msg)
            except UnicodeEncodeError:
                print(msg.encode("ascii", "replace").decode("ascii"))
            state["log_stream"].append(msg)
            
        log("[Coordination Agent] Synchronizing multilingual semantic overlay for live HUD.")
        
        diagnoses = state.get("diagnoses", [])
        if not diagnoses:
            log("[Coordination Agent] No diagnoses found to synchronize.")
            state["subtitle_overlay"] = {
                "en": "Status: Healthy / Monitoring",
                "hi": "स्थिति: स्वस्थ / निगरानी जारी",
                "mr": "स्थिती: निरोगी / निरीक्षण सुरू"
            }
            return state

        diag_data = diagnoses[0].get('diagnosis', {})
        conf = state.get("confidence_score", 0.0)
        
        # If it's a mock string from RAG instead of dictionary, handle it:
        if isinstance(diag_data, str):
            diag_str = diag_data
            status = "CONFIRMED" if conf > 0.7 else "INCONCLUSIVE"
            state["subtitle_overlay"] = {
                "en": f"Status: {status} - {diag_str}",
                "hi": "उपलब्ध नहीं",
                "mr": "उपलब्ध नाही"
            }
            return state
            
        # Standard translated extraction
        d_en = diag_data.get("diagnosis_en", diag_data.get("disease_name", diag_data.get("disease", "Unknown")))
        d_hi = diag_data.get("diagnosis_hi", "")
        d_mr = diag_data.get("diagnosis_mr", "")
        
        t_en = diag_data.get("treatment_en", diag_data.get("treatment", "No action needed."))
        t_hi = diag_data.get("treatment_hi", "")
        t_mr = diag_data.get("treatment_mr", "")
        
        status_en = "CONFIRMED" if conf > 0.7 else "INCONCLUSIVE"
        status_hi = "पुष्टि" if conf > 0.7 else "अनिर्णीत"
        status_mr = "निश्चित" if conf > 0.7 else "अनिर्णित"
        
        if conf < 0.3:
            status_en, status_hi, status_mr = "HEALTHY", "स्वस्थ", "निरोगी"
            d_en, d_hi, d_mr = "Monitoring", "निगरानी जारी", "निरीक्षण सुरू"
            t_en, t_hi, t_mr = "All clear", "सब ठीक है", "सर्व सुरक्षित"

        # Build final lightweight JSON payload strings
        sub_en = f"[{status_en}] {d_en} | Action: {t_en}"
        sub_hi = f"[{status_hi}] {d_hi} | उपाय: {t_hi}" if d_hi else f"[{status_en}] {d_en}"
        sub_mr = f"[{status_mr}] {d_mr} | उपाय: {t_mr}" if d_mr else f"[{status_en}] {d_en}"
        
        state["subtitle_overlay"] = {
            "en": sub_en,
            "hi": sub_hi,
            "mr": sub_mr
        }
        
        log("[Coordination Agent] Multisensory subtitle payload compiled successfully.")
        return state
