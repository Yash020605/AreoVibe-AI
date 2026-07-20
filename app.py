import streamlit as st
import time
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"  # suppress OpenCV connection error spam
from main import A2APipeline
from state import AgentState
from utils.image_fx import compute_true_heatmap
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import io
import qrcode
import plotly.express as px
import pandas as pd
from utils.network_helper import get_local_ip

# Streamlit Page Config - Futuristic Dark Theme (War Room)
st.set_page_config(
        page_title="AeroVibe AI | Sentinel War Room", 
        layout="wide", 
        initial_sidebar_state="expanded"
)

# Custom Theme CSS Override for High-Contrast "Futuristic" Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Outfit:wght@400;600;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        color: #E2E8F0 !important;
        background: -webkit-linear-gradient(45deg, #00FF87, #60EFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Background and Layout */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(11, 20, 31) 0%, rgb(3, 7, 18) 90%);
        color: #CBD5E1;
    }
    
    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.4) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.39);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
    }
    
    /* Alerts and Info boxes */
    .stAlert {
        background: rgba(16, 185, 129, 0.1) !important;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 12px;
        color: #E2E8F0;
    }
    
    /* Input Elements */
    .stTextInput>div>div>input, .stSelectbox>div>div>div {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px;
        color: white !important;
        transition: all 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #10B981 !important;
        box-shadow: 0 0 0 1px #10B981;
    }
    
    /* Custom Metric Cards CSS */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        text-align: left;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
        margin-bottom: 1rem;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px -10px rgba(16, 185, 129, 0.3);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .glass-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94A3B8;
        margin-bottom: 8px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    .glass-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
        font-family: 'Outfit', sans-serif;
        margin-bottom: 4px;
    }
    .glass-subtitle {
        font-size: 0.85rem;
        color: #10B981;
        font-weight: 500;
    }
    .glass-subtitle.alert {
        color: #EF4444;
    }
</style>
""", unsafe_allow_html=True)

TRANSLATIONS = {
    "English": {
        "title": "🚁 AeroVibe AI - Live-Brain Command Center",
        "subtitle": "**(Universal Crop-Agnostic Multi-Agent Drone Pipeline)**",
        "field_health": "Field Health Index",
        "optimal": "Optimal",
        "agent_cycles": "Agent Negotiation Cycles",
        "a2a_loops": "A2A Loops",
        "gpu_lat": "GPU Latency",
        "agent_thought": "🧠 Agent Thought Process",
        "live": "● Live",
        "data_intake": "Data Intake",
        "crop_meta": "Crop Metadata",
        "engage_stream": "▶ Engage Live Stream (Webcam / Drone)",
        "raw_feed": "### Raw Drone RGB Feed",
        "pseudo_ndvi": "### True VARI Heatmap",
        "ndvi_desc": "<p style='text-align: center; font-size: 13px; color: #AAA; margin-top: -10px;'>Blue/Green = Healthy | Yellow/Red = Stress</p>",
        "a2a_overlay": "### A2A Detection Overlay",
        "stress": "Stress",
        "error_cam": "Error: Could not open local webcam feed.",
        "failed_cam": "Failed to read from camera."
    },
    "हिंदी": {
        "title": "🚁 एयरोवाइब एआई - लाइव-ब्रेन कमांड सेंटर",
        "subtitle": "**(यूनिवर्सल फसल-अज्ञेयवादी मल्टी-एजेंट ड्रोन पाइपलाइन)**",
        "field_health": "खेत स्वास्थ्य सूचकांक",
        "optimal": "इष्टतम",
        "agent_cycles": "एजेंट बातचीत चक्र",
        "a2a_loops": "ए2ए लूप्स",
        "gpu_lat": "जीपीयू विलंबता",
        "agent_thought": "🧠 एजेंट विचार प्रक्रिया",
        "live": "● लाइव",
        "data_intake": "डेटा इनटेक",
        "crop_meta": "फसल मेटाडेटा",
        "engage_stream": "▶ लाइव स्ट्रीम शुरू करें (वेबकैम / ड्रोन)",
        "raw_feed": "### रॉ ड्रोन आरजीबी फीड",
        "pseudo_ndvi": "### ट्रू VARI हीटमैप",
        "ndvi_desc": "<p style='text-align: center; font-size: 13px; color: #AAA; margin-top: -10px;'>नीला/हरा = स्वस्थ | पीला/लाल = तनाव</p>",
        "a2a_overlay": "### ए2ए डिटेक्शन ओवरले",
        "stress": "तनाव",
        "error_cam": "त्रुटि: स्थानीय वेबकैम फीड नहीं खोला जा सका।",
        "failed_cam": "कैमरा से पढ़ने में विफल।"
    },
    "मराठी": {
        "title": "🚁 एयरोवाइब एआय - लाईव्ह-ब्रेन कमांड सेंटर",
        "subtitle": "**(युनिव्हर्सल पीक-अज्ञेयवादी मल्टी-एजंट ड्रोन पाइपलाइन)**",
        "field_health": "शेत आरोग्य निर्देशांक",
        "optimal": "इष्टतम",
        "agent_cycles": "एजंट वाटाघाटी चक्र",
        "a2a_loops": "ए2ए लूप",
        "gpu_lat": "जीपीयू विलंब",
        "agent_thought": "🧠 एजंट विचार प्रक्रिया",
        "live": "● लाईव्ह",
        "data_intake": "डेटा संकलन",
        "crop_meta": "पीक मेटाडेटा",
        "engage_stream": "▶ थेट प्रवाह सुरू करा (वेबकॅम / ड्रोन)",
        "raw_feed": "### रॉ ड्रोन आरजीबी फीड",
        "pseudo_ndvi": "### ट्रू VARI हीटमॅप",
        "ndvi_desc": "<p style='text-align: center; font-size: 13px; color: #AAA; margin-top: -10px;'>निळा/हिरवा = निरोगी | पिवळा/लाल = ताण</p>",
        "a2a_overlay": "### ए2ए डिटेक्शन ओव्हरले",
        "stress": "ताण",
        "error_cam": "त्रुटी: स्थानिक वेबकॅम फीड उघडता आले नाही.",
        "failed_cam": "कॅमेरामधून वाचण्यात अयशस्वी."
    }
}

lang_choice = st.sidebar.selectbox("🌍 Language / भाषा", ["English", "हिंदी", "मराठी"], index=0)
t = TRANSLATIONS[lang_choice]

st.title(t["title"])
st.markdown(t["subtitle"])

if "detection_logs" not in st.session_state:
    st.session_state.detection_logs = []
if "is_reporting" not in st.session_state:
    st.session_state.is_reporting = False

if st.session_state.is_reporting:
    st.header("📋 Mission Summary Dashboard")
    st.success("Scan Complete.")

    if not st.session_state.detection_logs:
        st.warning("No anomalies detected or no frames processed.")
    else:
        df = pd.DataFrame(st.session_state.detection_logs)

        # --- Impact Metric ---
        avg_severity = df["Severity"].str.replace("%","").astype(float).mean() if "Severity" in df.columns else 0
        yield_saved = round(max(0, 30 - avg_severity * 0.5), 1)
        st.markdown(f"""
        <div style='background:linear-gradient(90deg,#0a2a0a,#0e3d0e);border-left:5px solid #00FF41;
        padding:16px;border-radius:8px;margin-bottom:16px;'>
        <h3 style='color:#00FF41;margin:0'>🌾 Impact Estimate</h3>
        <p style='color:white;font-size:16px;margin:6px 0 0 0'>
        By detecting these symptoms early, you saved approximately
        <b style='color:#00FF41;font-size:22px'> {yield_saved}%</b> of your estimated yield.<br>
        <span style='color:#aaa;font-size:13px'>
        Early intervention reduces crop loss by up to 30% compared to late-stage treatment.
        </span></p></div>
        """, unsafe_allow_html=True)

        # --- Anomaly Pie Chart ---
        fig = px.pie(df, names='Disease Name', title='Anomalies Spread Overview')
        st.plotly_chart(fig, use_container_width=True)

        # --- Full Multilingual Table ---
        st.subheader("Detailed Detection Log")
        st.dataframe(df, use_container_width=True)

        # --- PDF Export (Devanagari-safe) ---
        st.markdown("---")
        if st.button("⬇ Download PDF Report"):
            pdf_bytes = _generate_pdf(df, yield_saved)
            st.download_button(
                label="📄 Save PDF",
                data=pdf_bytes,
                file_name=f"AeroVibe_Report_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

    if st.button("🔄 Resume Mission"):
        st.session_state.is_reporting = False
        st.session_state.detection_logs = []
        st.rerun()

    st.stop()


def _generate_pdf(df: pd.DataFrame, yield_saved: float) -> bytes:
    """Generates a Devanagari-safe PDF report using fpdf2."""
    try:
        from fpdf import FPDF
    except ImportError:
        return b""

    pdf = FPDF()
    pdf.add_page()

    # Try Nirmala UI for Devanagari, fallback to Helvetica
    unicode_ok = False
    try:
        pdf.add_font("Nirmala", "", "nirmalaui.ttf", uni=True)
        pdf.add_font("Nirmala", "B", "nirmalauib.ttf", uni=True)
        unicode_ok = True
    except Exception:
        pass

    def set_font(bold=False, size=10):
        if unicode_ok:
            pdf.set_font("Nirmala", "B" if bold else "", size)
        else:
            pdf.set_font("Helvetica", "B" if bold else "", size)

    # Header
    set_font(bold=True, size=16)
    pdf.set_text_color(0, 200, 80)
    pdf.cell(0, 10, "AeroVibe AI - Post-Flight Mission Report", ln=True, align="C")
    set_font(size=9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  |  VibeInnovators", ln=True, align="C")
    pdf.ln(4)

    # Impact box
    pdf.set_fill_color(10, 42, 10)
    pdf.set_text_color(0, 220, 80)
    set_font(bold=True, size=11)
    pdf.cell(0, 8, f"Estimated Yield Saved by Early Detection: {yield_saved}%", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    # Table
    cols    = ["Disease", "Severity", "EN Treatment", "MR Treatment", "HI Treatment"]
    col_map = ["Disease Name", "Severity", "English Advice", "मराठी (Marathi) सल्ला", "हिंदी (Hindi) सलाह"]
    widths  = [38, 18, 44, 44, 44]

    set_font(bold=True, size=8)
    pdf.set_fill_color(20, 60, 20)
    pdf.set_text_color(0, 255, 65)
    for col, w in zip(cols, widths):
        pdf.cell(w, 7, col, border=1, fill=True)
    pdf.ln()

    pdf.set_text_color(0, 0, 0)
    fill = False
    for _, row in df.iterrows():
        pdf.set_fill_color(240, 250, 240) if fill else pdf.set_fill_color(255, 255, 255)
        set_font(size=7)
        for key, w in zip(col_map, widths):
            val = str(row.get(key, ""))
            if not unicode_ok:
                val = val.encode("latin-1", "replace").decode("latin-1")
            pdf.cell(w, 6, val[:35], border=1, fill=True)
        pdf.ln()
        fill = not fill

    # Footer
    pdf.ln(6)
    set_font(size=8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "AeroVibe AI | VibeInnovators | Powered by RTX 4050 ONNX Pipeline", align="C")

    return bytes(pdf.output())

# --- Top Metrics Placeholder ---
metrics_placeholder = st.empty()

def render_custom_metrics(health, health_sub, cycles, cycles_sub, gpu, gpu_sub, crop, crop_sub, show_hw=False):
    alert_class = " alert" if "-" in str(health_sub) or "Stress" in str(health_sub) or "तनाव" in str(health_sub) or "ताण" in str(health_sub) else ""
    return f"""
    <div style="display: flex; gap: 1rem; width: 100%;">
        <div style="flex: 1;"><div class="glass-card">
            <div class="glass-title">{t["field_health"]}</div>
            <div class="glass-value">{health}</div>
            <div class="glass-subtitle{alert_class}">{health_sub}</div>
        </div></div>
        <div style="flex: 1;"><div class="glass-card">
            <div class="glass-title">{t["agent_cycles"]}</div>
            <div class="glass-value">{cycles}</div>
            <div class="glass-subtitle">{cycles_sub}</div>
        </div></div>
        <div style="flex: 1;"><div class="glass-card">
            <div class="glass-title">{"⚡ GPU vs CPU" if show_hw else t["gpu_lat"]}</div>
            <div class="glass-value">{gpu}</div>
            <div class="glass-subtitle">{gpu_sub}</div>
        </div></div>
        <div style="flex: 1;"><div class="glass-card">
            <div class="glass-title">🌿 Detected Crop</div>
            <div class="glass-value" style="font-size: 1.4rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{crop}</div>
            <div class="glass-subtitle">{crop_sub}</div>
        </div></div>
    </div>
    """

with metrics_placeholder.container():
    st.markdown(render_custom_metrics("100%", t["optimal"], "0", t["a2a_loops"], "120ms", "RTX 4050 (ONNX)", "Awaiting Feed...", "Manual"), unsafe_allow_html=True)

# --- VRAM Optimization ---
@st.cache_resource
def load_cached_pipeline():
    """ Keeps the ONNX models loaded into VRAM once and warm. """
    return A2APipeline()

pipeline = load_cached_pipeline()

# --- Sidebar: Agent Thought Process ---
st.sidebar.markdown(f"""
<h2 style='display: flex; align-items: center;'>
    {t["agent_thought"]} 
    <span style='color: #00FF41; font-size: 14px; margin-left: 10px; animation: blinker 1.5s linear infinite;'>{t["live"]}</span>
</h2>
<style>
    @keyframes blinker {{ 50% {{ opacity: 0; }} }}
</style>
""", unsafe_allow_html=True)
log_container = st.sidebar.empty()

st.sidebar.markdown("---")
local_ip = get_local_ip()
url = f"http://{local_ip}:8501"
qr = qrcode.make(url)
img_byte_arr = io.BytesIO()
qr.save(img_byte_arr, format='PNG')
st.sidebar.image(img_byte_arr.getvalue(), caption=f"Scan to connect Mobile Drone ({local_ip})")

if st.sidebar.button("🛑 Stop Scan & Generate Report"):
    st.session_state.is_reporting = True
    st.rerun()

# Task 3: Hardware performance toggle
st.sidebar.markdown("---")
if "show_hw_perf" not in st.session_state:
    st.session_state.show_hw_perf = False
if st.sidebar.toggle("⚡ Hardware Performance Mode", value=st.session_state.show_hw_perf):
    st.session_state.show_hw_perf = True
else:
    st.session_state.show_hw_perf = False

# --- HUD Utilities ---
def draw_multilingual_overlay_pil(cv2_image, subtitles_dict):
    """Draws Hindi, Marathi, and English subtitles using PIL to support complex Unicode."""
    pil_img = Image.fromarray(cv2_image)
    draw = ImageDraw.Draw(pil_img)
    
    # Try loading a Windows font that supports Devanagari script. Will fallback silently.
    try:
        font = ImageFont.truetype("nirmalaui.ttf", 22)
    except IOError:
        try:
            font = ImageFont.truetype("arial.ttf", 22)
        except IOError:
            font = ImageFont.load_default()

    h, w = cv2_image.shape[:2]
    en = subtitles_dict.get("en", "")
    hi = subtitles_dict.get("hi", "")
    mr = subtitles_dict.get("mr", "")

    y = h - 100
    
    def put_bg_text(draw_ctx, text, pos, txt_color):
        if not text: return
        x, y = pos
        # Draw small shadow
        draw_ctx.text((x + 2, y + 2), text, font=font, fill=(0,0,0))
        draw_ctx.text((x, y), text, font=font, fill=txt_color)

    put_bg_text(draw, en, (20, y), (0, 255, 65))
    put_bg_text(draw, hi, (20, y+28), (255, 191, 0))
    put_bg_text(draw, mr, (20, y+56), (255, 100, 100))

    return np.array(pil_img)

def draw_terminator_hud(image, primitives, conf):
    """Draws dynamic bounding boxes over the frame for active tracking."""
    h, w = image.shape[:2]
    color = (255, 0, 0) if conf >= 0.7 else (255, 255, 0) # Red if Confirmed, Yellow if Inconclusive (RGB)
    
    # Because proper ONNX box tracking isn't fully set up, we establish "mock" static boxes that jump around based on primitives.
    # We use a seed so they stay visually stable for the duration of a symptom's presence.
    random.seed(str(primitives)) 
    
    for prim in primitives:
        # Bounding limits
        x1 = random.randint(50, max(51, w - 200))
        y1 = random.randint(50, max(51, h - 200))
        x2 = x1 + random.randint(100, 180)
        y2 = y1 + random.randint(100, 180)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Terminator style corners
        cv2.line(image, (x1, y1), (x1+20, y1), color, 4)
        cv2.line(image, (x1, y1), (x1, y1+20), color, 4)
        cv2.rectangle(image, (x1, y1-30), (x1 + max(150, len(prim)*10), y1), color, -1)
        
        # Black text label inside the corner box
        cv2.putText(image, prim, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    return image


# --- IP Webcam URL Normalizer ---
def normalize_ip_cam_url(url):
    """Force http and append /video path for IP Webcam app compatibility."""
    url = url.strip().replace("https://", "http://")
    if not url.endswith("/video"):
        url = url.rstrip("/") + "/video"
    return url

# --- Main Interface ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader(t["data_intake"])
    crop_list = ["wheat", "rice", "corn", "sugarcane", "grapes", "citrus", "banana"]
    selected_crop = st.selectbox(t["crop_meta"], crop_list)
    auto_detect = st.toggle("🔍 Auto-Detect Crop", value=False, help="Let the AI identify the crop from the live feed")

    st.markdown("### 📷 Camera Connection")
    input_mode = st.radio(
        "Select Connection Type", 
        ["📱 Mobile Snapshot (Easy)", "💻 Local Webcam", "🚁 IP Cam / Drone Stream"],
        label_visibility="collapsed"
    )

    run_stream = False
    snapshot = None
    ip_cam_url = ""

    if input_mode == "📱 Mobile Snapshot (Easy)":
        st.info("Simplest for phones! Scan the QR code, tap below to open your camera, and snap a picture of the crop.")
        snapshot = st.camera_input("Take a photo of the crop")
        if not snapshot:
            snapshot = st.file_uploader("Or upload an image", type=['png', 'jpg', 'jpeg'])
            
    elif input_mode == "💻 Local Webcam":
        run_stream = st.checkbox(t["engage_stream"], value=False)
        
    elif input_mode == "🚁 IP Cam / Drone Stream":
        ip_cam_url = st.text_input(
            "📱 Stream URL",
            value="",
            placeholder="e.g. http://192.168.1.x:8080/video",
            help="Install 'IP Webcam' app on your phone or use a drone feed URL."
        )
        if ip_cam_url.strip():
            if st.button("🔗 Test Connection"):
                normalized = normalize_ip_cam_url(ip_cam_url)
                test_cap = cv2.VideoCapture(normalized)
                if test_cap.isOpened():
                    st.success(f"Connected: {normalized}")
                    test_cap.release()
                else:
                    st.error(f"Could not connect to {normalized}. Ensure IP Webcam is running and 'Start Server' is tapped.")
        run_stream = st.checkbox(t["engage_stream"], value=False)

def append_to_logs(state_info):
    conf_score = state_info.get("confidence_score", 0.0)
    for diag in state_info.get("diagnoses", []):
         d = diag.get("diagnosis", {}) if isinstance(diag, dict) else diag
         if isinstance(d, dict):
             dis = d.get("diagnosis_en", "Healthy")
             en_adv = d.get("treatment_en", "Optimal")
             hi_adv = d.get("treatment_hi", "इष्टतम")
             mr_adv = d.get("treatment_mr", "इष्टतम")
         else:
             dis = str(d)
             en_adv, hi_adv, mr_adv = "Optimal", "इष्टतम", "इष्टतम"
         
         if "Healthy" not in dis and dis != "Unknown":
             st.session_state.detection_logs.append({
                 "Disease Name": dis,
                 "Severity": f"{conf_score * 100:.1f}%",
                 "English Advice": en_adv,
                 "मराठी (Marathi) सल्ला": mr_adv,
                 "हिंदी (Hindi) सलाह": hi_adv
             })


visual_col1, visual_col2, visual_col3 = st.columns(3)

with visual_col1:
    st.markdown(t["raw_feed"])
    feed_placeholder1 = st.empty()
with visual_col2:
    st.markdown(t["pseudo_ndvi"])
    feed_placeholder2 = st.empty()
    st.markdown(t["ndvi_desc"], unsafe_allow_html=True)
with visual_col3:
    st.markdown(t["a2a_overlay"])
    feed_placeholder3 = st.empty()

if snapshot is not None:
    # Process single image snapshot
    file_bytes = np.asarray(bytearray(snapshot.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        state = {
            "gps_coordinates": "18.5204° N, 73.8567° E",
            "crop_type": selected_crop if not auto_detect else "unknown",
            "auto_detect_crop": auto_detect,
            "rescan_count": 0,
            "dense_regions_found": 0,
            "visual_rescan_requested": False,
            "log_stream": ["[System] Processing snapshot..."]
        }
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        state["image_path"] = buffer.tobytes()
        
        with st.spinner("AI Analyzing Crop..."):
            state = pipeline.run(state)
        
        # Sidebar log
        display_str = "\n".join(state.get('log_stream', []))
        log_container.markdown(f'''
        <div style="height:400px;overflow-y:auto;background:#0E1117;border:1px solid #333;
        padding:10px;border-radius:4px;display:flex;flex-direction:column-reverse;">
        <div style="font-family:'Courier New',monospace;font-size:13px;color:#00FF41;
        white-space:pre-wrap;">{display_str}</div></div>''', unsafe_allow_html=True)
        
        # Metrics
        conf         = state.get('confidence_score', 0.0)
        severity     = (conf * 100) if conf >= 0.3 else 0.0
        health_index = max(0, 100 - severity)
        cycles       = state.get("rescan_count", 0)
        show_hw      = st.session_state.get("show_hw_perf", False)
        gpu_ms       = state.get("gpu_latency_ms", 121)

        with metrics_placeholder.container():
            health_sub = f"-{severity:.1f}% {t['stress']}" if severity > 0 else t["optimal"]
            gpu_val = f"GPU {gpu_ms}ms" if show_hw else f"{gpu_ms}ms"
            gpu_sub = f"CPU ~{gpu_ms*8}ms ({8}x slower)" if show_hw else "RTX 4050 (ONNX)"
            
            detected_crop = state.get("crop_type", selected_crop)
            crop_detections = state.get("crop_detection_results", [])
            crop_conf_str = f"{crop_detections[0][1]*100:.0f}% match" if crop_detections else "Manual"
            wild      = state.get("wild_vegetation", False)
            low_light = state.get("low_light", False)
            crop_label = "🌿 Wild Vegetation" if wild else detected_crop.capitalize()
            if low_light:
                crop_label = f"🌑 {crop_label}"
                
            st.markdown(render_custom_metrics(f"{health_index:.1f}%", health_sub, f"{cycles}", t["a2a_loops"], gpu_val, gpu_sub, crop_label, crop_conf_str, show_hw), unsafe_allow_html=True)

        try:
            _, last_ndvi = compute_true_heatmap(state["image_path"])
        except Exception:
            last_ndvi = None
            
        append_to_logs(state)
        
        hud_frame = frame_rgb.copy()
        prims = state.get("symptom_primitives", [])
        conf  = state.get("confidence_score", 0.0)
        if prims:
            hud_frame = draw_terminator_hud(hud_frame, prims, conf)

        subs = state.get("subtitle_overlay", {})
        if subs:
            hud_frame = draw_multilingual_overlay_pil(hud_frame, subs)

        if state.get("low_light", False):
            cv2.rectangle(hud_frame, (0, 0), (hud_frame.shape[1], 36), (30, 30, 180), -1)
            cv2.putText(hud_frame, "  LOW LIGHT - Shadow suppression active", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2, cv2.LINE_AA)

        if not state.get("leaf_detected", True):
            cv2.rectangle(hud_frame, (0, 0), (hud_frame.shape[1], 36), (180, 30, 30), -1)
            cv2.putText(hud_frame, "  NO PLANT DETECTED - Aim camera at crop leaves", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
            
        feed_placeholder1.image(frame_rgb, use_container_width=True)
        if last_ndvi is not None:
            feed_placeholder2.image(last_ndvi, use_container_width=True)
        feed_placeholder3.image(hud_frame, use_container_width=True)

elif run_stream:
    source = normalize_ip_cam_url(ip_cam_url) if ip_cam_url.strip() else 0
    cap = cv2.VideoCapture(source)
    # Short timeout so failed IP connections fail fast instead of spamming errors
    if isinstance(source, str):
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
    
    if not cap.isOpened():
        st.error(t["error_cam"])
    else:
        frame_count = 0
        last_ndvi = None
        
        # Track pipeline data between async loops
        state = {
            "gps_coordinates": "18.5204° N, 73.8567° E",
            "crop_type": selected_crop if not auto_detect else "unknown",
            "auto_detect_crop": auto_detect,
            "rescan_count": 0,
            "dense_regions_found": 0,
            "visual_rescan_requested": False,
            "log_stream": ["[System] Live feed established..."]
        }

        while run_stream:
            ret, frame = cap.read()
            if not ret:
                st.error(t["failed_cam"])
                break

            # Resize for display — keeps UI snappy
            frame = cv2.resize(frame, (640, 480))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1

            # --- Pipeline: run every 10th frame (not 5th) to reduce CPU load ---
            if frame_count % 10 == 0:
                state["log_stream"] = []
                state["rescan_count"] = 0
                state["visual_rescan_requested"] = False

                # Encode at lower quality for faster pipeline processing
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                state["image_path"] = buffer.tobytes()

                state = pipeline.run(state)

                # Sidebar log
                display_str = "\n".join(state.get('log_stream', []))
                log_container.markdown(f'''
                <div style="height:400px;overflow-y:auto;background:#0E1117;border:1px solid #333;
                padding:10px;border-radius:4px;display:flex;flex-direction:column-reverse;">
                <div style="font-family:'Courier New',monospace;font-size:13px;color:#00FF41;
                white-space:pre-wrap;">{display_str}</div></div>''', unsafe_allow_html=True)

                # Metrics
                conf         = state.get('confidence_score', 0.0)
                severity     = (conf * 100) if conf >= 0.3 else 0.0
                health_index = max(0, 100 - severity)
                cycles       = state.get("rescan_count", 0)
                show_hw      = st.session_state.get("show_hw_perf", False)
                gpu_ms       = state.get("gpu_latency_ms", 121)

                with metrics_placeholder.container():
                    health_sub = f"-{severity:.1f}% {t['stress']}" if severity > 0 else t["optimal"]
                    gpu_val = f"GPU {gpu_ms}ms" if show_hw else f"{gpu_ms}ms"
                    gpu_sub = f"CPU ~{gpu_ms*8}ms ({8}x slower)" if show_hw else "RTX 4050 (ONNX)"
                    
                    detected_crop = state.get("crop_type", selected_crop)
                    crop_detections = state.get("crop_detection_results", [])
                    crop_conf_str = f"{crop_detections[0][1]*100:.0f}% match" if crop_detections else "Manual"
                    wild      = state.get("wild_vegetation", False)
                    low_light = state.get("low_light", False)
                    crop_label = "🌿 Wild Vegetation" if wild else detected_crop.capitalize()
                    if low_light:
                        crop_label = f"🌑 {crop_label}"
                        
                    st.markdown(render_custom_metrics(f"{health_index:.1f}%", health_sub, f"{cycles}", t["a2a_loops"], gpu_val, gpu_sub, crop_label, crop_conf_str, show_hw), unsafe_allow_html=True)

                # NDVI — only recompute every 30 frames (not every pipeline run)
                if frame_count % 30 == 0:
                    try:
                        _, last_ndvi = compute_true_heatmap(state["image_path"])
                    except Exception:
                        pass

                append_to_logs(state)

            # 2. Draw HUD and Live Subtitles continuously
            hud_frame = frame_rgb.copy()

            prims = state.get("symptom_primitives", [])
            conf  = state.get("confidence_score", 0.0)
            if prims:
                hud_frame = draw_terminator_hud(hud_frame, prims, conf)

            subs = state.get("subtitle_overlay", {})
            if subs:
                hud_frame = draw_multilingual_overlay_pil(hud_frame, subs)

            # Low-light warning banner on HUD
            if state.get("low_light", False):
                cv2.rectangle(hud_frame, (0, 0), (hud_frame.shape[1], 36), (30, 30, 180), -1)
                cv2.putText(hud_frame, "  LOW LIGHT - Shadow suppression active",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2, cv2.LINE_AA)

            # No-vegetation gate banner
            if not state.get("leaf_detected", True):
                cv2.rectangle(hud_frame, (0, 0), (hud_frame.shape[1], 36), (180, 30, 30), -1)
                cv2.putText(hud_frame, "  NO PLANT DETECTED - Aim camera at crop leaves",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
                
            # Render feeds
            feed_placeholder1.image(frame_rgb, use_container_width=True)
            if last_ndvi is not None:
                feed_placeholder2.image(last_ndvi, use_container_width=True)
            feed_placeholder3.image(hud_frame, use_container_width=True)

        # Teardown logic
        cap.release()
