import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import gdown
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

st.set_page_config(page_title="DermaScan AI", page_icon="microscope", layout="wide")

MODEL_PATH = "best_classical_model.pth"
FILE_ID = "1NyzHV2ny-J0cmVGGndLg_T_rh4XIbLla"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model (first time only)..."):
       gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False, fuzzy=True)
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

model = load_model()

label_map_reverse = {0: 'nv', 1: 'mel', 2: 'bkl', 3: 'bcc', 4: 'akiec', 5: 'vasc', 6: 'df'}
disease_names = {'nv': 'Melanocytic Nevus (harmless mole)', 'mel': 'Melanoma',
                 'bkl': 'Benign Keratosis', 'bcc': 'Basal Cell Carcinoma',
                 'akiec': 'Actinic Keratosis', 'vasc': 'Vascular Lesion', 'df': 'Dermatofibroma'}
risk_map = {'mel': 'high', 'bcc': 'high', 'akiec': 'medium', 'nv': 'low', 'bkl': 'low', 'vasc': 'low', 'df': 'low'}

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("DermaScan AI")
st.caption("Hybrid Quantum-Classical Skin Disease Diagnosis")

uploaded_file = st.file_uploader("Upload a skin lesion image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)

    input_tensor = eval_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
        pred_class = label_map_reverse[pred_idx.item()]

    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_idx.item())])[0, :]
    rgb_img = np.array(image.resize((224, 224))) / 255.0
    heatmap_vis = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.image(heatmap_vis, caption="Grad-CAM: Where the AI Looked", use_container_width=True)

    st.subheader(f"Prediction: {disease_names[pred_class]}")
    st.write(f"Confidence: **{confidence.item()*100:.1f}%**")

    risk = risk_map[pred_class]
    if risk == 'high':
        st.error("High Risk — Please see a dermatologist soon")
    else:
        st.success("Low Risk — Routine monitoring")