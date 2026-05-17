import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import gdown
import os
import pandas as pd

# Google Drive File ID and Model Path
GOOGLE_DRIVE_FILE_ID = "1reGBQyBks1mIMy05l5J7qUD0dslvN020"
MODEL_PATH = "checkpoint.pt"

# නිවැරදි වයස් කාණ්ඩ ලැයිස්තුව
AGE_GROUPS = [
    "0–2", "3–9", "10–19", "20–29",
    "30–39", "40–49", "50–59", "60–69", "70+"
]

# Page configuration
st.set_page_config(
    page_title="FairVision AI - Age Classification",
    layout="wide"
)

# Custom CSS Styling
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Title Styling */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 5px solid #4F46E5;
        margin-bottom: 1rem;
    }
    .metric-card-alt {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 5px solid #10B981;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.3rem;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #334155;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0.4rem;
    }
    </style>
""", unsafe_allow_html=True)  

# Model Architecture Definition
class FairVisionResNet(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Cached function to load model safely
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive... Please wait."):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

    model = FairVisionResNet(num_classes=9)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    
    state_dict = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    
    # 1. Clean DataParallel prefixes if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        cleaned_state_dict[name] = v
        
    # 2. Safe Loading: Filter out mismatched layers between server architecture and weights
    model_dict = model.state_dict()
    matched_state_dict = {k: v for k, v in cleaned_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    
    # 3. Update current model structure with downloaded valid weights
    model_dict.update(matched_state_dict)
    
    # Load weights safely
    model.load_state_dict(model_dict, strict=True)
    model.eval()
    return model

# Initialize Model
model = load_model()

# Image Transform Pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Prediction Function
def predict(image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
    return predicted.item(), confidence.item(), probabilities

# Sidebar Section
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.write("Upload an image below to analyze and predict the age group.")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Supports JPG, JPEG, and PNG formats."
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About FairVision")
    st.caption(
        "This system utilizes a fine-tuned ResNet-50 deep learning architecture "
        "to classify human faces into 9 distinct age cohorts safely and unbiasedly."
    )

# Main UI Panel
st.markdown('<div class="main-title">FairVision AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Age Group Classification Framework</div>', unsafe_allow_html=True)

if uploaded_file:
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    # Left Column: Image Display
    with col1:
        st.markdown('<div class="section-header">Analyzed Image</div>', unsafe_allow_html=True)
        image = Image.open(uploaded_file).convert("RGB")
        # මෙතන තිබුණු channels="RGB" ඉවත් කර දෝෂය නිවැරදි කර ඇත
        st.image(
            image, 
            use_container_width=True
        )
        
    # Right Column: Results Display
    with col2:
        with st.spinner(" Running Deep Learning Inference..."):
            pred_idx, confidence, probs = predict(image)
        predicted_label = AGE_GROUPS[pred_idx]
        
        st.markdown('<div class="section-header">Classification Results</div>', unsafe_allow_html=True)
        
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Age Group</div>
                    <div class="metric-value">{predicted_label} Yrs</div>
                </div>
            """, unsafe_allow_html=True)
            
        with metric_col2:
            st.markdown(f"""
                <div class="metric-card-alt">
                    <div class="metric-label">Confidence Score</div>
                    <div class="metric-value">{confidence * 100:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown('<div class="section-header"> Top 3 Probabilities</div>', unsafe_allow_html=True)
        top_probs, top_indices = torch.topk(probs, 3)
        
        for i in range(3):
            label = AGE_GROUPS[top_indices[i].item()]
            score = top_probs[i].item()
            
            p_col1, p_col2 = st.columns([3, 1])
            p_col1.write(f"**{label}**")
            p_col2.write(f"`{score * 100:.2f}%`")
            st.progress(score)
            
        st.markdown('<div class="section-header">Full Distribution</div>', unsafe_allow_html=True)
        df = pd.DataFrame({
            "Age Group": AGE_GROUPS,
            "Probability (%)": [p.item() * 100 for p in probs]
        })
        st.bar_chart(df.set_index("Age Group"), color="#4F46E5")

else:
    st.info(" Please upload an image from the sidebar panel to begin the classification process.")