import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="CIFAR-10 Classifier", layout="centered")

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load("cifar_model.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()

# ---------------------------
# Classes
# ---------------------------
classes = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

# ---------------------------
# Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize(40),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# ---------------------------
# Header UI
# ---------------------------
st.markdown("""
    <h1 style='text-align: center;'>CIFAR-10 Image Classifier</h1>
    <p style='text-align: center; color: gray;'>
        Upload an image and the model will predict one of 10 classes
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg"]
)

# ---------------------------
# Prediction
# ---------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("### Preview")
    st.image(image, width=350)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)

    label = classes[predicted.item()]
    confidence = conf.item() * 100

    st.markdown("---")

    # Styled result box
    st.markdown(f"""
    <div style="
        padding: 20px;
        border-radius: 12px;
        background-color: #E8F5E9;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    ">
        Prediction: {label} <br>
        Confidence: {confidence:.2f}%
    </div>
    """, unsafe_allow_html=True)

    if confidence < 50:
        st.warning("Low confidence prediction. Try simpler images.")
