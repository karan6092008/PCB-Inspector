import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
st.set_page_config(
    page_title="PCB Defect Inspector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1rem; color: #888; text-align: center; margin-bottom: 1.5rem; }
    .stImage > img { border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
    </style>
""", unsafe_allow_html=True)
st.markdown("<div class='main-header'> PCB Defect Inspector AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Automated Quality Control powered by YOLOv8</div>", unsafe_allow_html=True)
st.sidebar.title("Controls")
conf_threshold = st.sidebar.slider(
    "Detection Confidence", 
    min_value=0.05, 
    max_value=1.0, 
    value=0.25, 
    step=0.05
)
st.sidebar.markdown("---")
st.sidebar.subheader("Input Selection")
uploaded_file = st.sidebar.file_uploader("Upload custom PCB image", type=["jpg", "jpeg", "png"])
demo_folder = "demo_samples"
demo_files = []
if os.path.exists(demo_folder):
    demo_files = [f for f in os.listdir(demo_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
selected_demo = st.sidebar.selectbox("Or choose a pre-loaded sample:", ["None"] + demo_files)
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.sidebar.success("Loaded custom uploaded image.")
elif selected_demo != "None":
    image_path = os.path.join(demo_folder, selected_demo)
    image = Image.open(image_path)
    st.sidebar.info(f"Loaded demo sample: {selected_demo}")
@st.cache_resource
def load_model():
    return YOLO("weights/best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    model = None
if image is not None and model is not None:
    # Responsive Columns: Side-by-side on desktop, auto-stacked vertically on mobile
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Input PCB")
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("### Detection Output")
        
        # Run YOLO Inference
        results = model.predict(image, conf=conf_threshold)
        res_plotted = results[0].plot()  # Render bounding box overlay
        
        st.image(res_plotted, use_container_width=True)
    st.markdown("---")
    boxes = results[0].boxes
    defect_count = len(boxes)
    if defect_count > 0:
        st.error(f" **{defect_count} Defect(s) Detected!**")
        with st.expander(" View Defect Breakdown", expanded=True):
            class_names = model.names
            detected_classes = [class_names[int(cls)] for cls in boxes.cls]
            counts = {name: detected_classes.count(name) for name in set(detected_classes)}
            m_cols = st.columns(len(counts))
            for idx, (defect, count) in enumerate(counts.items()):
                m_cols[idx].metric(label=defect.title(), value=f"{count} found")
    else:
        st.success(" **PCB Passed Quality Control: No Defects Detected.**")

else:
    # Initial Prompt Screen
    st.info(" Select a demo sample or upload a PCB image from the sidebar to begin inspection.")