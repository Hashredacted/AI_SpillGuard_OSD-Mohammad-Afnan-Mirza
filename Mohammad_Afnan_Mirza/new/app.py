import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import io
import warnings
import os

# ============ Suppress Warnings ============
warnings.filterwarnings("ignore")

# ============ Page Setup ============
st.set_page_config(
    page_title="🛢️ AI SpillGuard",
    page_icon="🛢️",
    layout="wide",
)

# ============ Custom CSS Styling ============
st.markdown("""
<style>
/* Global Styling */
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    background: #0b111a;
    color: #f5f7fa;
}

/* Header Gradient */
h1 {
    background: linear-gradient(90deg, #00b4d8, #48cae4, #90e0ef);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    text-align: center;
}

/* Subheaders */
h2, h3, h4 {
    color: #caf0f8 !important;
    font-weight: 600;
}

/* Section Divider */
hr {
    border: 0;
    height: 1px;
    background: linear-gradient(to right, #0077b6, #00b4d8, transparent);
    margin: 1.5rem 0;
}

/* Metric Cards */
div[data-testid="stMetricValue"] {
    color: #90e0ef;
    font-weight: 700;
}

div[data-testid="stMetricLabel"] {
    color: #adb5bd;
}

/* Buttons */
div.stDownloadButton > button {
    background: linear-gradient(90deg, #0077b6, #00b4d8);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    transition: all 0.3s ease-in-out;
}

div.stDownloadButton > button:hover {
    background: linear-gradient(90deg, #00b4d8, #48cae4);
    transform: scale(1.03);
}

/* File Uploader */
div.stFileUploader > div > div > div > button {
    background: linear-gradient(90deg, #00b4d8, #48cae4);
    border: none;
    color: white;
    font-weight: 600;
}

/* Expander */
.streamlit-expanderHeader {
    background-color: rgba(0, 180, 216, 0.15);
    border-radius: 6px;
}

/* Image Frames */
img {
    border-radius: 12px;
    box-shadow: 0px 0px 8px rgba(0,0,0,0.3);
}

/* Columns */
[data-testid="column"] {
    padding: 1rem;
}

/* Table Styling */
[data-testid="stDataFrame"] {
    border: 1px solid #1a2a40;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ============ Constants ============
COLOR_MAP = [
    [0, 0, 0],       # Background
    [0, 255, 255],   # Oil Spill
    [255, 0, 0],     # Vegetation
    [153, 76, 0],    # Ship
    [0, 153, 0],     # Land
]
CLASS_NAMES = ["Background", "Oil Spill", "Vegetation", "Ship", "Land"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "unet_model.keras")


# ============ Load Model ============
@st.cache_resource
def load_model_cached():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        return None


# ============ Helper Functions ============
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_resized = np.array(image.resize((256, 256)))
    image_normalized = image_resized / 255.0
    return np.expand_dims(image_normalized, axis=0), image_resized


def postprocess_mask(prediction):
    mask = np.argmax(prediction, axis=-1)
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(COLOR_MAP):
        colored_mask[mask == class_idx] = color
    return mask[0], colored_mask[0]


def create_simple_overlay(original, mask):
    mask_pil = Image.fromarray(mask)
    original_resized = original.resize((256, 256))
    overlay = Image.blend(original_resized.convert("RGBA"), mask_pil.convert("RGBA"), alpha=0.5)
    return overlay.convert("RGB")


def pil_to_bytes(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return img_byte_arr


# ============ Main App ============
def main():
    st.markdown("<h1>🛢️ AI SpillGuard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color:#adb5bd;'>Deep Learning–powered oil spill detection and segmentation from satellite imagery.</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    model = load_model_cached()
    if model is None:
        st.stop()

    uploaded_file = st.file_uploader("📤 Upload a satellite image", type=["jpg", "jpeg", "png", "tif", "tiff"])

    if uploaded_file:
        try:
            image = Image.open(uploaded_file)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("🌍 Original Image")
                st.image(image, use_container_width=True)

            with st.spinner("🔍 Processing image..."):
                processed, image_resized = preprocess_image(image)
                prediction = model.predict(processed, verbose=0)
                class_mask, colored_mask = postprocess_mask(prediction)

            mask_pil = Image.fromarray(colored_mask)
            overlay = create_simple_overlay(image, colored_mask)

            with col2:
                st.subheader("🗺️ Segmentation Mask")
                st.image(mask_pil, use_container_width=True)
                st.subheader("🧩 Overlay")
                st.image(overlay, use_container_width=True)

            # ===== Analysis =====
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("📈 Analysis")

            total_pixels = class_mask.size
            class_percentages = [(np.sum(class_mask == i) / total_pixels) * 100 for i in range(5)]
            oil_percent = class_percentages[1]

            colA, colB, colC = st.columns(3)
            colA.metric("🧮 Total Pixels", f"{total_pixels:,}")
            colB.metric("🌊 Oil Coverage", f"{oil_percent:.2f}%")

            if oil_percent > 5:
                colC.error("🚨 Major oil spill detected!")
            elif oil_percent > 1:
                colC.warning("⚠️ Moderate spill detected")
            elif oil_percent > 0.1:
                colC.info("ℹ️ Minor spill detected")
            else:
                colC.success("✅ No significant spill")

            # ===== Class Distribution =====
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("🧭 Class Distribution")

            class_data = []
            for idx, (class_name, color) in enumerate(zip(CLASS_NAMES, COLOR_MAP)):
                pixels = np.sum(class_mask == idx)
                percentage = class_percentages[idx]
                class_data.append({
                    "Class": class_name,
                    "Pixels": f"{pixels:,}",
                    "Percentage": f"{percentage:.2f}%",
                })

            metric_cols = st.columns(5)
            for idx, data in enumerate(class_data):
                with metric_cols[idx]:
                    color_array = np.full((25, 25, 3), COLOR_MAP[idx], dtype=np.uint8)
                    st.image(color_array, use_container_width=True)
                    st.metric(data["Class"], data["Percentage"], help=f"{data['Pixels']} pixels")

            with st.expander("📋 Detailed Table"):
                df = pd.DataFrame(class_data)
                st.dataframe(df, use_container_width=True)

            # ===== Downloads =====
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("💾 Download Results")

            colD, colE = st.columns(2)
            with colD:
                st.download_button(
                    label="📥 Download Mask",
                    data=pil_to_bytes(mask_pil),
                    file_name="segmentation_mask.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with colE:
                st.download_button(
                    label="📥 Download Overlay",
                    data=pil_to_bytes(overlay),
                    file_name="segmentation_overlay.png",
                    mime="image/png",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("Try uploading a different image format (JPEG or PNG preferred).")


if __name__ == "__main__":
    main()
