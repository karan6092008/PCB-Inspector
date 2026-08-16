import streamlit as st
from ultralytics import YOLO
from PIL import Image
from src.utils import draw_custom_bboxes

st.set_page_config(page_title="PCB Defect Inspector", layout="wide")
st.title("Automated PCB Defect Inspector")
@st.cache_resource
def load_yolo_model():
    return YOLO("weights/best.pt")
try:
    model= load_yolo_model()
except Exception:
    st.error("Model weights not found in 'weights/best.pt'. Please run 'python src/train.py' first.")
    st.stop()
st.sidebar.header("Settings")
conf_thresh= st.sidebar.slider("Confidence Threshold",0.1,1.0,0.40)
uploaded_file=st.sidebar.file_uploader("Upload PCB Image", type=["jpg","png","jpeg"])

if uploaded_file:
    col1, col2= st.columns(2)
    image= Image.open(uploaded_file).convert("RGB")
    with col1:
        st.subheader("Source PCB")
        st.image(image, use_container_width=True)
    results= model(image, conf=conf_thresh)[0]
    res_plotted = results.plot()
    with col2:
        st.subheader("AI Analysis Result")
        st.image(res_plotted, use_container_width=True)
    st.divider()
    boxes= results.boxes
    if len(boxes) >0:
        st.error(f"**{len(boxes)} Defects Detected!**")
        defects = [model.names[int(b.cls[0])] for b in boxes]
        cols = st.columns(len(set(defects)))
        for idx, defect_type in enumerate(set(defects)):
            cols[idx].metric(label=defect_type, value=defects.count(defect_type))
    else:
        st.success("**PCB Passed Quality Control: No Defects Detected.**")