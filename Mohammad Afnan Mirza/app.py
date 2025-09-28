import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_unet():
    return tf.keras.models.load_model("unet_model.keras")

unet_model = load_unet()

# Get model input shape dynamically (batch, H, W, C)
input_shape = unet_model.input_shape[1:3]  # (H, W)
IMG_SIZE = (input_shape[1], input_shape[0])  # (width, height) for PIL

st.title("🌊 Oil Spill Detection & Segmentation")
st.write("Upload an image → classify (Spill / No Spill) → generate segmentation mask.")

# ==============================
# Threshold slider
# ==============================
threshold_percent = st.slider(
    "Spill Threshold (% of pixels)", 
    min_value=1, 
    max_value=60, 
    value=25, 
    step=1, 
    help="Minimum % of pixels predicted as spill to classify as SPILL"
) / 100.0  # convert % to fraction

# ==============================
# Upload Image
# ==============================
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ==============================
    # Preprocess
    # ==============================
    img_resized = image.resize(IMG_SIZE)  # must match model input
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)  # (1, H, W, C)

    # ==============================
    # Segmentation
    # ==============================
    pred_mask = unet_model.predict(img_batch)[0]

    if pred_mask.shape[-1] == 1:  # binary segmentation
        pred_mask = (pred_mask[..., 0] > 0.5).astype(np.uint8)
    else:  # multi-class segmentation
        pred_mask = np.argmax(pred_mask, axis=-1).astype(np.uint8)

    # ==============================
    # Classification (based on mask ratio)
    # ==============================
    total_pixels = pred_mask.size
    spill_pixels = np.sum(pred_mask)
    spill_ratio = spill_pixels / total_pixels

    if spill_ratio > threshold_percent:
        label = "Spill"
    else:
        label = "No Spill"

    st.subheader("🟢 Classification Result")
    st.write(f"**Prediction:** {label}")
    st.write(f"Spill ratio: {spill_ratio*100:.2f}% (threshold: {threshold_percent*100:.2f}%)")

    # ==============================
    # Show Segmentation
    # ==============================
    if label == "Spill":
        st.subheader("🟡 Segmentation Mask")
        st.image(pred_mask * 255, caption="Predicted Spill Region", use_column_width=True)
    else:
        st.info("No spill detected → skipping segmentation mask.")
