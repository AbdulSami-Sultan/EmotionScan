import os
import cv2
import torch
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from torchvision import transforms
from your_ddamfn_model import DDAMFNPlusPlus  # Import your model architecture

app = Flask(__name__)
socketio = SocketIO(app)

# Load the trained model
model = DDAMFNPlusPlus()  # Initialize the model
model.load_state_dict(torch.load('ddamfn++.pth', map_location=torch.device('cpu')))  # Load trained weights
model.eval()  # Set the model to evaluation mode

# Define image preprocessing (same as during training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to the input size of the model
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# List of emotions (assuming DDAMFN++ outputs these classes)
emotion_classes = ['Neutral', 'Happy', 'Sad', 'Angry', 'Surprised']

# Function to preprocess the frame before feeding it to the model
def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (as the model expects RGB images)
    img = transform(img)  # Apply the same transformations used during training
    img = img.unsqueeze(0)  # Add batch dimension
    return img

# Function for generating frames and detecting emotions
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Preprocess the frame and pass it through the trained model
        img_tensor = preprocess_frame(frame)

        with torch.no_grad():
            outputs = model(img_tensor)  # Forward pass through the model
            _, predicted = torch.max(outputs, 1)  # Get the predicted emotion label
            emotion_label = emotion_classes[predicted.item()]  # Convert prediction to class label

        # Draw the predicted emotion on the video frame
        cv2.putText(frame, f'Mood: {emotion_label}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Send the detected emotion to the front-end via WebSocket
        socketio.emit('mood_update', {'mood': emotion_label})

        # Encode the frame to JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Return the frame in a format Flask can stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route to stream the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)
