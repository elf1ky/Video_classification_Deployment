import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from torchvision import transforms
from django.shortcuts import render
import cv2
import numpy as np
import os, tempfile
from PIL import Image

# ==============================================
# 1️⃣  Model Setup
# ==============================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["Non-Shoplifter", "Shoplifter"]  # make sure order matches your training
NUM_FRAMES = 32
RESIZE = 112  # or whatever RESIZE you used

# Initialize model
model = r3d_18(weights=None)  # same as your training (you fine-tuned pretrained model)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load("detector/r3d_video_classification_model.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ==============================================
# 2️⃣  Transform (same as training)
# ==============================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RESIZE, RESIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45],
                         std=[0.225, 0.225, 0.225]),
])

# ==============================================
# 3️⃣  Frame Extraction Function
# ==============================================
def extract_frames(video_path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(frame)
            frames.append(frame)
    cap.release()

    if len(frames) < num_frames:
        # pad with copies of the last frame if too short
        while len(frames) < num_frames:
            frames.append(frames[-1])

    video_tensor = torch.stack(frames)  # shape: (T, C, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # -> (C, T, H, W)
    return video_tensor.unsqueeze(0).to(DEVICE)  # add batch dim

# ==============================================
# 4️⃣  Prediction Function
# ==============================================
def predict(video_tensor):
    with torch.no_grad():
        outputs = model(video_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)
        label = CLASSES[predicted.item()]
        confidence = probs[0][predicted.item()].item() * 100
    return label, confidence

# ==============================================
# 5️⃣  Django View
# ==============================================
def home(request):
    prediction = None
    confidence = None

    if request.method == "POST" and request.FILES.get("video"):
        video_file = request.FILES["video"]

        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            for chunk in video_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        # Extract frames & predict
        frames = extract_frames(tmp_path)
        label, conf = predict(frames)

        # Clean up
        os.remove(tmp_path)

        prediction = label
        confidence = f"{conf:.2f}%"

    return render(request, "home.html", {"prediction": prediction, "confidence": confidence})
