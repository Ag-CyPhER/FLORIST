# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 17:44:45 2025

@author: puran
"""

# Importing libraries
import numpy as np
import torch
from PIL import Image
import os
from torchvision import transforms
from tqdm.notebook import tqdm
from joblib import load
from PIL import ImageFile
import json
from joblib import dump
import pandas as pd
from dinov2.dinov2.models import vision_transformer as vits

# CODE FOR EXTRACTING FEATURE EMBEDDINGS FROM Full DATASET
cwd = os.getcwd()

# Defining path to the images included in the Full dataset
# Flower and Non flower each for each species
ROOT_DIR = os.path.join(cwd, "Embed check/Full")

labels = {}

# Reading images
for folder in os.listdir(ROOT_DIR):
    print(folder)
    for file in os.listdir(os.path.join(ROOT_DIR, folder)):
        if file.endswith(".jpg") or file.endswith("jpeg"):
            full_name = os.path.join(ROOT_DIR, folder, file)
            labels[full_name] = folder

files = labels.keys()

# Extracting feature embeddings using Dinov2 - vision foundation model
# dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14") # 21 M
# dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14') # 86 M
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14') # 300 M
# dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14') # 1100 M

# Running code on gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2.to(device)

# Transforming images into format acceptable by Dinov2 model
transform_image = transforms.Compose([
                                    transforms.ToTensor(), 
                                    transforms.Resize((840, 840)), 
                                    transforms.Normalize([0.5], [0.5])
                                    ])                      
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Function to load images from the training dataset
def load_image(img: str) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    try:
        img = Image.open(img)
        transformed_img = transform_image(img)[:3].unsqueeze(0)
        return transformed_img
    except Exception as e:
        print(f"Error loading image {img}: {e}")
        return None

# Function to compute feature embeddings from images in the training dataset    
def compute_embeddings(files: list) -> dict:
    """
    Create an index that contains all of the images in the specified list of files.
    """
    all_embeddings = {}
    
    with torch.no_grad():
      for i, file in enumerate(tqdm(files)):
        embeddings = dinov2(load_image(file).to(device))

        all_embeddings[file] = np.array(embeddings[0].cpu().numpy()).reshape(1, -1).tolist()

    with open("all_embeddings.json", "w") as f:
        f.write(json.dumps(all_embeddings))

    return all_embeddings

# Compute embeddings and store them in a DataFrame
embeddings = compute_embeddings(files)

# Feature embeddings of the images in the training dataset
embedding_list = list(embeddings.values())

full_embeddings = np.array(embedding_list).reshape(-1, 1024) # dinov2_vitl14

# Load the trained Linear SVC model 
clf = load('svc_model.joblib')

# Use it to make predictions on the full dataset
y_pred = clf.predict(full_embeddings)

# Load only the sheet named "Full dataset" from the Excel file
df = pd.read_excel('Flower classification project.xlsx', sheet_name='Full dataset')

# Creating a new column for saving the predictions into a new excel file
df['Predicted_Label'] = y_pred

# Save the updated DataFrame to a new Excel file
df.to_excel('Flower_classification_with_predictions.xlsx', index=False)