import cv2
import torch
import torchvision.transforms as transforms
from ddamfn_model import DDAMFNPlusPlus
import logging
from PIL import Image

# Setting up the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_emotion_live(model_path, device):
    logger.info("Starting real-time emotion detection...")
    
    # Load model
    model = DDAMFNPlusPlus(num_classes=7).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Error: Unable to access the webcam.")
        return

    emotions = ["Neutral", "Happy", "Sad", "Angry", "Surprised", "Fear", "Disgust"]

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Error: Unable to read from the webcam.")
            break

        # Preprocess frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        # Predict emotion
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            emotion = emotions[predicted.item()]

        # Display emotion on frame
        cv2.putText(frame, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Real-time detection stopped.")

