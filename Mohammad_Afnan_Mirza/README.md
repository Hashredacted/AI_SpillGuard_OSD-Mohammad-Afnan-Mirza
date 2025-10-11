# 🛢️ AI SpillGuard: Oil Spill Detection & Segmentation

## 📌 Project Overview

**AI SpillGuard** is an advanced deep learning system for automated detection and segmentation of oil spills from satellite and aerial imagery. Built on a sophisticated **U-Net architecture**, this solution provides pixel-level segmentation to accurately identify and quantify oil spill extents while distinguishing between multiple environmental classes.

![AI SpillGuard](https://img.shields.io/badge/AI-Oil_Spill_Detection-blue)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-TensorFlow-orange)
![Web App](https://img.shields.io/badge/Web_App-Streamlit-red)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB RAM minimum
- 2GB storage space

### Installation & Run
```bash
# Clone the repository
git clone <repository-url>
cd AI_SpillGuard

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run app.py
```

Access the web interface at: <br><br>
[![Open in Streamlit](https://img.shields.io/badge/Open%20in-Streamlit-FF4B4B?logo=streamlit&style=for-the-badge)](https://aispillguardosd-mohammad-afnan-mirza-ab7rbxck5ehj68knalfegk.streamlit.app/)

## 📁 Project Structure

```
AI_SpillGuard/Mohammad_Afnan_Mirza/
│
├── 📊 new                     # Developer workspace
│   ├── U_Net.ipynb             # U-Net model implementation
│   ├── app.py                  # Streamlit application
│   ├── requirements.txt        # Dependencies
│   └── unet_model.keras        # Trained model weights
│
├── 📓 oil detection.ipynb      # previous training notebook
├── 📋 requirements.txt         # previous Project dependencies
├── 🎯 unet_model.keras         # Pre-trained model
├── 📄 LICENSE                  # Project license
└── 📖 README.md               # This file
```

## 🎯 Key Features

### 🔍 Advanced Detection Capabilities
- **Multiclass Semantic Segmentation**: Background, Oil Spill, Vegetation, Ship, and Land
- **Real-time Processing**: Instant analysis with sub-second inference times
- **Quantitative Analytics**: Pixel-level area calculations and spill severity classification
- **Interactive Visualization**: Side-by-side comparison with overlay options
- **Export Functionality**: Download masks, overlays, and analysis reports

### 📊 Performance Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| **Overall Accuracy** | ~97% | Pixel-wise classification |
| **Oil Spill Precision** | 0.95 | Low false positives |
| **Recall** | 0.95 | High detection rate |
| **Inference Time** | <500ms | Real-time capability |

## 🛠️ Technical Implementation

### 🧠 Model Architecture
The system uses a **U-Net architecture** with the following components:
- **Encoder**: Feature extraction with convolutional layers
- **Bottleneck**: Context capture at reduced resolution
- **Decoder**: Upsampling with skip connections for precise localization
- **Output**: 5-class segmentation with softmax activation

### 🎨 Class Mapping
| Class | Label | Color | Description |
|-------|-------|-------|-------------|
| Background | 0 | Black | Open water, clouds |
| Oil Spill | 1 | Cyan | Hydrocarbon slicks |
| Vegetation | 2 | Red | Coastal vegetation |
| Ship | 3 | Brown | Vessels, platforms |
| Land | 4 | Green | Terrestrial areas |

## 📊 Dataset Information

Kaggle Dataset [![Kaggle](https://img.shields.io/badge/Kaggle-Explore-blue?logo=kaggle)](https://www.kaggle.com/datasets/nabilsherif/oil-spill)

### 🗂️ Dataset Structure
The project uses the **Oil Spill Detection Dataset** containing:
- 1000+ satellite images with pixel-level annotations
- Multiple spill scenarios and environmental conditions
- Balanced distribution across different classes

### 🔧 Preprocessing Pipeline
1. **Resizing**: Standardized to 256×256 pixels
2. **Normalization**: Pixel values scaled to [0, 1] range
3. **Augmentation**: Rotation, flipping, brightness variations
4. **Validation**: 70-15-15 split for training, validation, and testing

## 🚀 Usage Guide

### Web Application
1. **Launch the app**: Run `streamlit run app.py`
2. **Upload Image**: Use the file uploader for satellite imagery
3. **View Results**: Examine segmentation masks and overlays
4. **Analyze Metrics**: Check spill percentage and severity
5. **Export**: Download results for reporting

### Code Integration
```python
# Load the pre-trained model
model = load_model("unet_model.keras")

# Preprocess input image
processed_image = preprocess_image(your_image)

# Get predictions
predictions = model.predict(processed_image)

# Post-process results
mask, colored_mask = postprocess_mask(predictions)
```

## 📈 Model Training

### Training Configuration
```python
# Key hyperparameters
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
LOSS_FUNCTION = 'categorical_crossentropy'
OPTIMIZER = 'adam'
```

### Evaluation Metrics
- **Accuracy**: Overall pixel classification
- **IoU (Intersection over Union)**: Segmentation quality
- **Precision & Recall**: Detection performance
- **Dice Coefficient**: Similarity measure

## 🚨 Alert System

### Severity Classification
| Spill Percentage | Level | Action |
|------------------|-------|--------|
| < 0.1% | 🟢 Normal | No action |
| 0.1-1% | 🔵 Minor | Monitor |
| 1-5% | 🟠 Significant | Investigate |
| > 5% | 🔴 Major | Immediate response |

## 🔮 Future Enhancements

### Short-term Goals
- [ ] Multi-temporal analysis for spill tracking
- [ ] Integration with satellite data APIs
- [ ] Enhanced model explainability
- [ ] Batch processing capabilities

### Long-term Vision
- [ ] 3D spill volume estimation
- [ ] Weather condition integration
- [ ] Global monitoring network
- [ ] Predictive modeling

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/hashredacted/ai_spillguard_osd-mohammad-afnan-mirza/blob/main/LICENSE) file for details.

## 👨‍💻 Developer

**Mohammad Afnan Mirza**
- Deep Learning Engineer & Environmental AI Enthusiast
- Focused on computer vision applications for environmental monitoring

## 🌍 Environmental Impact

This project supports:
- **SDG 14**: Life Below Water - Marine ecosystem protection
- **SDG 9**: Industry Innovation - Advanced monitoring systems
- **Early Intervention**: Rapid response to environmental disasters

---

<div align="center">

### **Protecting Our Oceans Through AI** 🌊

*"Early detection saves ecosystems"*

**⭐ If this project helps protect our environment, give it a star!**

</div>

## 📞 Support

For questions or support:
- Create an issue in the repository
- Check the documentation in the notebooks
- Review the example implementations

---

*Last updated: October 2025*





