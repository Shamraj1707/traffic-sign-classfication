from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import torch
from PIL import Image
from torchvision import transforms
from torch import nn

app = Flask(__name__)

# Path to save uploaded images
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the traffic sign classification model
model = nn.Sequential(
    nn.Conv2d(3, 16, (2, 2), (1, 1), padding='same'),
    nn.BatchNorm2d(16),
    nn.ReLU(True),
    nn.MaxPool2d((2, 2)),
    nn.Conv2d(16, 32, (2, 2), (1, 1), padding='same'),
    nn.BatchNorm2d(32),
    nn.ReLU(True),
    nn.MaxPool2d((2, 2)),
    nn.Conv2d(32, 64, (2, 2), (1, 1), padding='same'),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.MaxPool2d((2, 2)),
    nn.Flatten(),
    nn.Linear(64 * 8 * 8, 256),
    nn.ReLU(True),
    nn.Linear(256, 43)
)

# Load the model's state dictionary
model.load_state_dict(torch.load('traffic_sign_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Assuming your class labels are:
class_labels = ['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 'No passing', 'No passing veh over 3.5 tons', 'Right-of-way at intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles', 'Veh > 3.5 tons prohibited', 'No entry', 'General caution', 'Dangerous curve left', 'Dangerous curve right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 'End speed + passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing', 'End no passing veh > 3.5 tons']  # Your actual class labels here

# Route to render the HTML form
@app.route('/')
def upload_form():
    return render_template('index.html', output='')  # Initialize output as empty

# Route to handle file upload and classification
@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Open the image
        image = Image.open(filepath)
        
        # Apply the same transformations as used during training
        img_tensor = transform(image).unsqueeze(0)
        
        # Perform prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            class_id = predicted.item()
        
        # Get the class label
        result = class_labels[class_id]
        
        # Render the result back into the textarea
        return render_template('index.html', output=result)

if __name__ == "__main__":
    app.run(debug=True)
