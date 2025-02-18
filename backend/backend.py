from fastapi import FastAPI, UploadFile, File
import cv2
import torch
import numpy as np
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import smtplib
from email.message import EmailMessage
from io import BytesIO
from starlette.middleware.cors import CORSMiddleware
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import os

# Inicializar FastAPI
app = FastAPI()

# Permitir CORS para frontend local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), "emotion_detections.pt")
device = torch.device("cpu")  # Forzar uso de CPU
model = EfficientNet.from_name("efficientnet-b1")  # Modelo más ligero para CPU
in_features = model._fc.in_features
model._fc= torch.nn.Sequential(
      torch.nn.Dropout(0.5),
      torch.nn.Linear(model._fc.in_features, 5)
)
state_dict = torch.load(MODEL_PATH, map_location=device)
if "model" in state_dict:
    state_dict = state_dict["model"]

model.load_state_dict(state_dict, strict=False)

# Verificar que los pesos se aplicaron correctamente
for name, param in model.named_parameters():
    print(f"🔹 {name}: {param.shape}") 

model.eval().to(device)

# Etiquetas de emociones
emotions = ['angry', 'fear', 'happy', 'sad', 'surprise']

# Transformaciones de imagen
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((64, 64)),  # Reducir resolución para mejorar velocidad en CPU
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image).unsqueeze(0).to(device)

# Función para enviar email
def send_email(emotion):
    sender_email = "hillarygy07@gmail.com"
    receiver_email = "fvalenciat2@gmail.com"
    password = "ziah wxpo ahma fgkn"
    subject = "Alerta de Emoción Detectada"
    message = f"Se detectó la emoción: {emotion} en el video en tiempo real."
    
    msg = EmailMessage()
    msg.set_content(message)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, password)
        server.send_message(msg)
    print(f"📩 Email enviado: {emotion}")

# Ruta para procesar imágenes
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("L")
    image_np = np.array(image)
    
    # Detección de rostro con OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(image_np, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return JSONResponse(content={"error": "No se detectó un rostro"}, status_code=400)
    
    # Procesar y predecir emoción
    img_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        confidence, pred_index = torch.max(probabilities, 0)
        predicted_emotion = emotions[pred_index.item()]
        
        # Enviar email si es enojo, miedo o tristeza
        if predicted_emotion in ['angry', 'fear', 'sad']:
            send_email(predicted_emotion)
    
    return {"emotion": predicted_emotion, "confidence": float(confidence), "faces": faces.tolist()}