import streamlit as st
import cv2
import numpy as np
from roboflow import Roboflow
from PIL import Image
from dotenv import load_dotenv
import os

# -----------------------------
# CONFIG & LOAD ENV
# -----------------------------
load_dotenv()
API_KEY = os.getenv("API")
WORKSPACE = os.getenv("WORKSPACE")
PROJECT_NAME = os.getenv("PROJECT")
MODEL_VERSION = int(os.getenv("MODEL_VERSION", 2))

SIMILARITY_TOLERANCE = 0.08  # 8% size difference allowed
MAX_DISPLAY_WIDTH = 900

st.set_page_config(layout="wide", page_title="Screw Similarity Finder")
st.title("🔩 Screw & Nut Similarity Finder")

# -----------------------------
# SESSION STATE
# -----------------------------
if "reference" not in st.session_state:
    st.session_state.reference = None
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0


# -----------------------------
# LOAD MODEL (Hosted API)
# -----------------------------
@st.cache_resource
def load_roboflow_model():
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace(WORKSPACE).project(PROJECT_NAME)
        return project.version(MODEL_VERSION).model
    except Exception as e:
        st.error(f"Failed to connect to Roboflow: {e}")
        return None


model = load_roboflow_model()


# -----------------------------
# UTILS
# -----------------------------
def get_bbox_dimensions(obj):
    # Max is usually length, min is usually diameter/thickness
    return max(obj['width'], obj['height']), min(obj['width'], obj['height'])


def normalized_distance(a, b):
    return abs(a - b) / max(a, b) if max(a, b) > 0 else 0


# -----------------------------
# UI & UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload image of hardware", type=["jpg", "png", "jpeg"])

if uploaded_file and model:
    # 1. Process Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize for display and sync
    h_orig, w_orig = image.shape[:2]
    scale = MAX_DISPLAY_WIDTH / w_orig if w_orig > MAX_DISPLAY_WIDTH else 1
    image = cv2.resize(image, (int(w_orig * scale), int(h_orig * scale)))
    h, w = image.shape[:2]

    # 2. Run Inference
    with st.spinner("AI analyzing hardware..."):
        cv2.imwrite("temp_infer.jpg", image)
        result = model.predict("temp_infer.jpg", confidence=55).json()
        detections = result.get("predictions", [])

    # 3. Draw Preview with NAMES
    preview = image.copy()
    for obj in detections:
        x1 = int(obj['x'] - obj['width'] / 2)
        y1 = int(obj['y'] - obj['height'] / 2)
        x2 = int(obj['x'] + obj['width'] / 2)
        y2 = int(obj['y'] + obj['height'] / 2)

        # Draw Box
        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Draw Label
        label = f"{obj['class']}"
        cv2.putText(preview, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 4. DRAWING CANVAS
    st.subheader("🖍️ Step 1: Draw a box over the Master object")
    from streamlit_drawable_canvas import st_canvas

    canvas = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)),
        height=h, width=w, drawing_mode="rect",
        key=f"canvas_{st.session_state.canvas_key}",
    )

    # 5. SELECT REFERENCE LOGIC
    if canvas.json_data and canvas.json_data.get("objects"):
        if st.button("➕ Set as Master Reference"):
            rect = canvas.json_data["objects"][-1]
            rx1, ry1 = rect["left"], rect["top"]
            rx2, ry2 = rx1 + rect["width"], ry1 + rect["height"]

            best_overlap, selected_obj = 0, None
            for obj in detections:
                x1, y1 = obj['x'] - obj['width'] / 2, obj['y'] - obj['height'] / 2
                x2, y2 = obj['x'] + obj['width'] / 2, obj['y'] + obj['height'] / 2

                # Intersection math
                inter = max(0, min(rx2, x2) - max(rx1, x1)) * max(0, min(ry2, y2) - max(ry1, y1))
                if inter > best_overlap:
                    best_overlap = inter
                    selected_obj = obj

            if selected_obj:
                l, t = get_bbox_dimensions(selected_obj)
                st.session_state.reference = {
                    "class": selected_obj['class'],
                    "length": l,
                    "thickness": t
                }
                st.success(f"Reference set to: {selected_obj['class']}")
                st.session_state.canvas_key += 1
                st.rerun()

    # 6. MATCHING LOGIC
    if st.session_state.reference:
        ref = st.session_state.reference
        match_img = image.copy()
        match_count = 0

        for obj in detections:
            # Must be same class (e.g., don't match a screw to a nut)
            if obj['class'] == ref['class']:
                l, t = get_bbox_dimensions(obj)

                # Check if dimensions are within 8% tolerance
                if normalized_distance(l, ref['length']) <= SIMILARITY_TOLERANCE and \
                        normalized_distance(t, ref['thickness']) <= SIMILARITY_TOLERANCE:
                    match_count += 1
                    x1 = int(obj['x'] - obj['width'] / 2)
                    y1 = int(obj['y'] - obj['height'] / 2)
                    x2 = int(obj['x'] + obj['width'] / 2)
                    y2 = int(obj['y'] + obj['height'] / 2)

                    cv2.rectangle(match_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(match_img, "MATCH", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        st.success(f"✅ Found {match_count} items matching your reference!")
        st.image(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))

# RESET BUTTON
if st.button("🔄 Reset All"):
    st.session_state.reference = None
    st.session_state.canvas_key += 1
    st.rerun()