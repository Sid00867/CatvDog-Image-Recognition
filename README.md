# CatvDog-Image-Recognition

A CNN built using **PyTorch** to classify images of dogs and cats. The model is trained on a labeled dataset with multiple convolutional and pooling layers, followed by fully connected layers for final binary classification. Training was done on Google Colab using a **T4 GPU**.

## Files

- **`cat_dog_model_weights.pth`** – Trained model parameters.  
- **`cnn_predict`** – CLI tool.  
  Use the following command to make a prediction:  
  `python cnn_predict --image_url <IMAGE_URL>`
- **`predictu_GUI`** – Simple Tkinter-based GUI for prediction.  
  Paste an image URL and click **"Predict"**.

## Dataset

The model was trained on the [Cat and Dog dataset from Kaggle](https://www.kaggle.com/datasets/tongpython/cat-and-dog).  

## Example

**CLI:**  
`python cnn_predict --image_url https://example.com/dog.jpg`

**GUI:**  
Run:  
`python predictu_GUI.py`  
Then paste the image URL and click **"Predict"**.
