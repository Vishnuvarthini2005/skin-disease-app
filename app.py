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

st.set_page_config(page_title="DermaScan AI", page_icon="🔬", layout="wide")

MODEL_PATH = "best_classical_model.pth"
FILE_ID = "1NyzHV2ny-J0cmVGGndLg_T_rh4XIbLla"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model (first time only)..."):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)