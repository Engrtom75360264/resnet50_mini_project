import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ---------------------------------------------------
# 1. Load Model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    checkpoint = torch.load("save_model/resnet50_cifar100_checkpoint.pth", map_location="cpu")

    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 100)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    classes = checkpoint["classes"]
    return model, classes

model, classes = load_model()

# ---------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 for ResNet-50
    transforms.ToTensor(),          # Convert to tensor [0,1]
    transforms.Normalize(           # Normalize like ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(img):
    # Apply preprocessing
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)
    return classes[predicted.item()]

# ---------------------------------------------------
# 3. Streamlit UI
# ---------------------------------------------------
st.title("🐯 CIFAR-100 Image Classifier (ResNet-50)")
st.subheader("📤 Upload Your Own Image")

upload_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if upload_file:
    # Read image in memory
    img = Image.open(upload_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict class
    pred = predict(img)
    st.success(f"Predicted Class: **{pred}**")
