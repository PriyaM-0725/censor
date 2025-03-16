import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from torchvision.models.segmentation import deeplabv3_resnet101

# Load Pretrained DeepLabV3 Model
model = deeplabv3_resnet101(pretrained=True).eval()

# Function to calculate clothing percentage
def detect_clothing(image_path):
    image = Image.open(image_path).convert("RGB")
    
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    output_predictions = output.argmax(0).byte().numpy()

    # Classes 15-18 correspond to clothing (DeepLabV3 categories)
    skin_pixels = np.sum(output_predictions == 0)  # Background/Skin
    cloth_pixels = np.sum((output_predictions >= 15) & (output_predictions <= 18))

    total_pixels = skin_pixels + cloth_pixels
    if total_pixels == 0:
        return 100  # Assume full coverage if total pixels are zero
    cloth_percentage = (cloth_pixels / total_pixels) * 100
    return cloth_percentage

# Function to open file dialog and process image
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    # Detect clothing percentage
    cloth_percentage = detect_clothing(file_path)
    result = "Minimal Clothing (Nude)" if cloth_percentage < 30 else "Fully Clothed"

    # Display the image
    img = Image.open(file_path)
    img = img.resize((300, 300))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

    # Show results
    result_label.config(text=f"Clothing Coverage: {cloth_percentage:.2f}%\nResult: {result}")

    # Show popup message
    messagebox.showinfo("Clothing Detection Result", f"Clothing Coverage: {cloth_percentage:.2f}%\nResult: {result}")

# Create Tkinter UI
root = tk.Tk()
root.title("Clothing Detection")
root.geometry("400x500")
root.configure(bg="white")

title_label = tk.Label(root, text="Clothing Detection System", font=("Arial", 14, "bold"), bg="white")
title_label.pack(pady=10)

image_label = tk.Label(root, bg="white")
image_label.pack()

upload_button = tk.Button(root, text="Upload Image", command=upload_image, font=("Arial", 12), bg="lightblue")
upload_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12), bg="white")
result_label.pack(pady=10)

# Run Tkinter
root.mainloop()
