import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import requests
from io import BytesIO
import torch
from cnn import CatDogCNN
from torchvision import transforms
import base64
import re

# Load the model once at startup
model = CatDogCNN()
weights_path = "./cat_dog_model_weights.pth"
try:
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True))
except Exception as e:
    messagebox.showerror("Model Load Error", f"Could not load model weights: {e}")
    exit()
model.eval()

# Define the same transforms as training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_image_from_input(input_str):
    """
    Accepts either a URL or a data URI (data:image/...) and returns a PIL Image.
    """
    if input_str.startswith("data:image"):
        # Parse the data URI
        match = re.match(r'data:(image/\w+);base64,(.*)', input_str)
        if not match:
            raise ValueError("Invalid data URI format.")
        image_data = match.group(2)
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"Could not decode image from data URI: {e}")
    else:
        # Assume it's a URL
        try:
            response = requests.get(input_str)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"Could not load image from URL: {e}")

def predict_from_url():
    url = url_entry.get()
    if not url:
        messagebox.showwarning("Input Error", "Please enter an image URL or data URI.")
        return

    try:
        image = load_image_from_input(url)
    except Exception as e:
        messagebox.showerror("Image Error", str(e))
        return

    # Show the image in the GUI
    img_resized = image.resize((128, 128))
    img_tk = ImageTk.PhotoImage(img_resized)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # Preprocess and predict
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    probability = output.item()
    prediction = 'Dog' if probability >= 0.5 else 'Cat'
    confidence = probability if prediction == 'Dog' else 1 - probability

    result_label.config(
        text=f"Prediction: {prediction}\nConfidence: {confidence:.2%}"
    )

# Build the GUI
root = tk.Tk()
root.title("Cat vs Dog Classifier")

tk.Label(root, text="Enter Image URL or data URI:").pack(pady=5)
url_entry = tk.Entry(root, width=60)
url_entry.pack(pady=5)

predict_btn = tk.Button(root, text="Predict", command=predict_from_url)
predict_btn.pack(pady=5)

image_label = tk.Label(root)
image_label.pack(pady=5)

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()