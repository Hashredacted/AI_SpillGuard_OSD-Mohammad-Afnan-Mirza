import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import io
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Oil Spill Segmentation",
    page_icon="🛢️",
    layout="wide"
)

# Constants
COLOR_MAP = [
    [0, 0, 0], [0, 255, 255], [255, 0, 0], [153, 76, 0], [0, 153, 0]
]

CLASS_NAMES = ["Background", "Oil Spill", "Ship", "Land", "Vegetation"]

@st.cache_resource
def load_model_cached():
    """Load the trained U-Net model"""
    try:
        return load_model("unet_model.keras") #replace with actual path
    except:
        st.error("Could not load model. Ensure 'unet_model.keras' is in the directory.")
        return None

def preprocess_image(image):
    """Preprocess image for model input"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize and normalize
    image_resized = np.array(image.resize((256, 256)))
    image_normalized = image_resized / 255.0
    return np.expand_dims(image_normalized, axis=0), image_resized

def postprocess_mask(prediction):
    """Convert model prediction to mask"""
    mask = np.argmax(prediction, axis=-1)
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    for class_idx, color in enumerate(COLOR_MAP):
        colored_mask[mask == class_idx] = color
    
    return mask[0], colored_mask[0]

def create_simple_overlay(original, mask):
    """Simple overlay using PIL"""
    if isinstance(mask, np.ndarray):
        mask_pil = Image.fromarray(mask)
    else:
        mask_pil = mask
    
    # Resize original to match mask size (256x256)
    original_resized = original.resize((256, 256))
    
    # Create overlay by blending
    overlay = Image.blend(original_resized.convert('RGBA'), 
                         mask_pil.convert('RGBA'), 
                         alpha=0.5)
    
    return overlay.convert('RGB')

def pil_to_bytes(image):
    """Convert PIL image to bytes for download"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

def main():
    st.title("🛢️ Oil Spill Segmentation")
    st.write("Upload a satellite image to detect oil spills and other features.")
    
    # Model loading
    model = load_model_cached()
    if model is None:
        return
    
    # File upload
    uploaded_file = st.file_uploader("Choose image", type=['jpg', 'jpeg', 'png', 'tif', 'tiff'])
    
    if uploaded_file is not None:
        try:
            # Load image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            # Process image
            with st.spinner("Analyzing image..."):
                processed, image_resized = preprocess_image(image)
                prediction = model.predict(processed, verbose=0)
                class_mask, colored_mask = postprocess_mask(prediction)
            
            # Convert to PIL images for display and download
            mask_pil = Image.fromarray(colored_mask)
            overlay = create_simple_overlay(image, colored_mask)
            
            # Display results
            with col2:
                st.subheader("Segmentation Mask")
                st.image(mask_pil, use_container_width=True)
                
                st.subheader("Overlay")
                st.image(overlay, use_container_width=True)
            
            # Analysis
            st.subheader("Analysis")
            total_pixels = class_mask.size
            
            # Calculate all class percentages
            class_percentages = []
            for i in range(5):
                class_pixels = np.sum(class_mask == i)
                percentage = (class_pixels / total_pixels) * 100
                class_percentages.append(percentage)
            
            oil_percent = class_percentages[1]
            
            # Overall metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Pixels", f"{total_pixels:,}")
            col2.metric("Oil Spill Coverage", f"{oil_percent:.2f}%")
            
            with col3:
                if oil_percent > 5:
                    st.error("🚨 Major oil spill detected!")
                elif oil_percent > 1:
                    st.warning("⚠️ Significant oil spill detected")
                elif oil_percent > 0.1:
                    st.info("ℹ️ Minor oil spill detected")
                else:
                    st.success("✅ No significant oil spill detected")
            
            # Detailed class breakdown
            st.subheader("Detailed Class Analysis")
            
            # Create a clean table for class distribution
            class_data = []
            for idx, (class_name, color) in enumerate(zip(CLASS_NAMES, COLOR_MAP)):
                pixels = np.sum(class_mask == idx)
                percentage = class_percentages[idx]
                class_data.append({
                    'Class': class_name,
                    'Pixels': f"{pixels:,}",
                    'Percentage': f"{percentage:.2f}%",
                    'Color': f"rgb({color[0]}, {color[1]}, {color[2]})"
                })
            
            # Display as metrics in columns
            st.write("**Class Distribution:**")
            metric_cols = st.columns(5)
            for idx, data in enumerate(class_data):
                with metric_cols[idx]:
                    # Create color swatch
                    color_array = np.full((30, 30, 3), COLOR_MAP[idx], dtype=np.uint8)
                    st.image(color_array, use_container_width=True)
                    st.metric(
                        label=data['Class'],
                        value=data['Percentage'],
                        help=f"{data['Pixels']} pixels"
                    )
            
            # Optional: Display as table
            with st.expander("View Detailed Table"):
                df = pd.DataFrame(class_data)
                st.dataframe(df[['Class', 'Pixels', 'Percentage']], use_container_width=True)
            
            # Download options
            st.subheader("Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                mask_bytes = pil_to_bytes(mask_pil)
                st.download_button(
                    "📥 Download Mask",
                    data=mask_bytes,
                    file_name="segmentation_mask.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with col2:
                overlay_bytes = pil_to_bytes(overlay)
                st.download_button(
                    "📥 Download Overlay", 
                    data=overlay_bytes,
                    file_name="segmentation_overlay.png",
                    mime="image/png",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Try uploading a different image format (JPEG or PNG recommended)")

if __name__ == "__main__":

    main()

