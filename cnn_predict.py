import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
from cnn import CatDogCNN
import requests
from io import BytesIO

def predict_image(model, image_url):
    """
    Function to load an image from a URL, preprocess it, and get a prediction from the model.
    """
    try:
        # Download the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')

        # Define the same transformations as used in training
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Apply the transformations
        input_tensor = transform(image)
        # Add a batch dimension (the model expects a batch of images)
        input_batch = input_tensor.unsqueeze(0)

        # Make a prediction
        with torch.no_grad(): # No need to calculate gradients for inference
            output = model(input_batch)

        # The output is a probability. 0 for 'cat', 1 for 'dog'.
        probability = output.item()
        prediction = 'Dog' if probability >= 0.5 else 'Cat'
        confidence = probability if prediction == 'Dog' else 1 - probability

        # Print the results
        print("\n" + "="*30)
        print(f"      PREDICTION RESULT")
        print("="*30)
        print(f"URL:        {image_url}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2%}")
        print("="*30)

    except requests.exceptions.RequestException as e:
        print(f"Error: Could not retrieve the image from URL '{image_url}': {e}")
    except Exception as e:
        print(f"An error occurred while processing the image: {e}")


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Test a Cat/Dog CNN on a single image from a URL.")
    parser.add_argument('--image_url', type=str, required=True, help="URL to the image you want to test.")
    args = parser.parse_args()

    # --- Load the Model and Weights ---
    model = CatDogCNN()
    weights_path = "./cat_dog_model_weights.pth"  # Hardcoded path to the weights file
    try:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True))
        # print(f"Successfully loaded weights from {weights_path}")
    except FileNotFoundError:
        print(f"Error: The weights file was not found at '{weights_path}'")
        exit() # Exit the script if weights can't be loaded
    except Exception as e:
        print(f"An error occurred loading the model weights: {e}")
        exit()

    # Set the model to evaluation mode
    model.eval()
    predict_image(model, args.image_url)
