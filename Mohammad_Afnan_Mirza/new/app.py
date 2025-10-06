import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import io
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# Streamlit page setup
st.set_page_config(
    page_title="🛢️ AI SpillGuard – Oil Spill Segmentation",
    page_icon="🛢️",
    layout="wide",
)

# =========================
# 🔧 Constants
# =========================
COLOR_MAP = [
    [0, 0, 0],       # Background - Black
    [0, 255, 255],   # Oil Spill - Cyan
    [255, 0, 0],     # Vegetation - Red
    [153, 76, 0],    # Ship - Brown
    [0, 153, 0],     # Land - Green
]
CLASS_NAMES = ["Background", "Oil Spill", "Vegetation", "Ship", "Land"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "unet_model.keras")

# =========================
# 🧠 Load Model
# =========================
@st.cache_resource
def load_model_cached():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        return None


# =========================
# 🧩 Helper Functions
# =========================
def preprocess_image(image):
    """Resize and normalize image for model input."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    image_resized = np.array(image.resize((256, 256)))
    image_normalized = image_resized / 255.0
    return np.expand_dims(image_normalized, axis=0), image_resized


def postprocess_mask(prediction):
    """Convert model output to colored segmentation mask."""
    mask = np.argmax(prediction, axis=-1)
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for class_idx, color in enumerate(COLOR_MAP):
        colored_mask[mask == class_idx] = color

    return mask[0], colored_mask[0]


def create_simple_overlay(original, mask):
    """Blend mask with original image for visualization."""
    mask_pil = Image.fromarray(mask)
    original_resized = original.resize((256, 256))
    overlay = Image.blend(original_resized.convert("RGBA"), mask_pil.convert("RGBA"), alpha=0.5)
    return overlay.convert("RGB")


def pil_to_bytes(image):
    """Convert PIL Image to bytes for downloading."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return img_byte_arr


# =========================
# 🚀 App Logic
# =========================
def main():
    st.markdown("<h1 style='text-align: center;'>🛢️ AI SpillGuard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>A deep learning system for oil spill segmentation in satellite imagery.</p>", unsafe_allow_html=True)
    st.divider()

    model = load_model_cached()
    if model is None:
        st.stop()

    uploaded_file = st.file_uploader("📤 Upload a satellite image (JPG, PNG, or TIFF)", type=["jpg", "jpeg", "png", "tif", "tiff"])

    if uploaded_file:
        try:
            image = Image.open(uploaded_file)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("🌍 Original Image")
                st.image(image, use_container_width=True)

            with st.spinner("🔍 Analyzing image... Please wait."):
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

            # =========================
            # 📊 Analysis Section
            # =========================
            st.divider()
            st.subheader("📈 Analysis")

            total_pixels = class_mask.size
            class_percentages = [(np.sum(class_mask == i) / total_pixels) * 100 for i in range(5)]
            oil_percent = class_percentages[1]

            colA, colB, colC = st.columns(3)
            colA.metric("🧮 Total Pixels", f"{total_pixels:,}")
            colB.metric("🌊 Oil Spill Coverage", f"{oil_percent:.2f}%")

            if oil_percent > 5:
                colC.error("🚨 Major oil spill detected!")
            elif oil_percent > 1:
                colC.warning("⚠️ Significant oil spill detected")
            elif oil_percent > 0.1:
                colC.info("ℹ️ Minor oil spill detected")
            else:
                colC.success("✅ No significant oil spill detected")

            # =========================
            # 🧾 Class Breakdown
            # =========================
            st.divider()
            st.subheader("🧭 Class Distribution")

            class_data = []
            for idx, (class_name, color) in enumerate(zip(CLASS_NAMES, COLOR_MAP)):
                pixels = np.sum(class_mask == idx)
                percentage = class_percentages[idx]
                class_data.append({
                    "Class": class_name,
                    "Pixels": f"{pixels:,}",
                    "Percentage": f"{percentage:.2f}%",
                    "Color": f"rgb({color[0]}, {color[1]}, {color[2]})"
                })

            metric_cols = st.columns(5)
            for idx, data in enumerate(class_data):
                with metric_cols[idx]:
                    color_array = np.full((25, 25, 3), COLOR_MAP[idx], dtype=np.uint8)
                    st.image(color_array, use_container_width=True)
                    st.metric(data["Class"], data["Percentage"], help=f"{data['Pixels']} pixels")

            with st.expander("📋 View Detailed Table"):
                df = pd.DataFrame(class_data)
                st.dataframe(df[["Class", "Pixels", "Percentage"]], use_container_width=True)

            # =========================
            # 💾 Download Section
            # =========================
            st.divider()
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
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("Try uploading a different image format (JPEG or PNG recommended).")


if __name__ == "__main__":
    main()
