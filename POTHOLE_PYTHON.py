#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def calculate_pothole_area(pred_mask):
    # Calculate the total number of pixels in the mask
    total_pixels = pred_mask.size

    # Calculate the number of pixels that are part of the pothole (i.e., where the mask is 1)
    pothole_pixels = np.sum(pred_mask)

    # Calculate the percentage of the image covered by potholes
    pothole_percentage = (pothole_pixels / total_pixels) * 100

    return pothole_percentage

def predict_and_show(model, image_path, output_path):
    # Read the image from the provided path
    image = cv2.imread(image_path)

    # Check if the image was loaded correctly
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Resize image to match the input size of the model (assuming it's 400x400)
    image_resized = cv2.resize(image, (400, 400))

    # Normalize the image to range [0, 1]
    image_norm = image_resized / 255.0

    # Add batch dimension (model expects a batch of images)
    image_input = np.expand_dims(image_norm, axis=0)

    # Predict the mask using the model
    pred_mask = model.predict(image_input)

    # Threshold the predicted mask to obtain binary values
    pred_mask = (pred_mask > 0.2).astype(np.uint8)

    # Save the mask as an image
    mask_image = (pred_mask.squeeze() * 255).astype(np.uint8)  # Convert to 0-255 range
    cv2.imwrite(output_path, mask_image)

    # Calculate the area covered by potholes in the predicted mask
    pothole_percentage = calculate_pothole_area(pred_mask.squeeze())

    # Display the original image and the predicted mask
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_resized)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask.squeeze(), cmap='gray')  # Squeeze to remove batch dimension
    plt.title("Predicted Segmentation Mask")
    plt.axis('off')

    plt.show()

    # Print the coverage percentage
    print(f"Pothole coverage: {pothole_percentage:.2f}%")

def main():
    # Path to the saved model
    model_path = "pothole_segmentation_model2.h5"  # Update with the actual model path
    model = load_model(model_path)

    # Input image path
    input_image_path = "path_to_input_image.png"  # Update with the actual input image path

    # Output mask image path
    output_mask_path = "output_mask.png"  # The path where the output mask will be saved

    # Perform prediction and show the results
    predict_and_show(model, input_image_path, output_mask_path)

if __name__ == "__main__":
    main()

