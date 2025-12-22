# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 21:06:36 2025

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

# CODE FOR EXTRACTING FEATURE EMBEDDINGS FROM TRAINING DATASET
cwd = os.getcwd()

# Defining path to the images included in the training dataset - this includes 40 images for
# Flower and Non flower each for each species
ROOT_DIR = os.path.join(cwd, "Embed check/Training")

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

# Ground truth labels
y = [labels[file] for file in files]

# Feature embeddings of the images in the training dataset
embedding_list = list(embeddings.values())

# Saving feature embeddings of all the images in flower classification dataset
np.save('Embed check/feature_embeddings.npy', embedding_list)

train_emb = np.array(embedding_list).reshape(-1, 1024) # dinov2_vitl14
# train_emb = np.array(embedding_list).reshape(-1, 768) # dinov2_vitb14

# Storing training embeddings in an excel file 
df = pd.DataFrame(train_emb)
df.to_excel('Embed check/Training.xlsx')



# CODE FOR EXTRACTING FEATURE EMBEDDINGS FROM TESTING DATASET
# TESTING
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# Load and preprocess the test dataset
test_labels = {}  # Dictionary to hold test file paths and their labels

# Testing dataset includes 1000 images randomly selected from the entire flower classification dataset 
# excluding the ones included in the training dataset (Independent dataset)
TEST_DIR = os.path.join(cwd, "Embed check/Testing")

# Reading images from Testing dataset
for folder in os.listdir(TEST_DIR):
    for file in os.listdir(os.path.join(TEST_DIR, folder)):
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            full_name = os.path.join(TEST_DIR, folder, file)
            test_labels[full_name] = folder     

test_files = list(test_labels.keys())

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Extract embeddings for the test dataset
test_embeddings = []
test_y = []

# Extracting feature embeddings of images from the Testing dataset
# We will use the trained SVC model to perform flower classifications using teh feature embeddings from testing dataset
# Extract embeddings with a progress bar
with torch.no_grad():
    for file in tqdm(test_files, desc="Processing Images", unit="image"):
        img_tensor = load_image(file)
        if img_tensor is None:  # Skip corrupted images
            continue
        embedding = dinov2(img_tensor.to(device))
        test_embeddings.append(np.array(embedding[0].cpu()).reshape(1024))  # Flatten to 1D
        test_y.append(test_labels[file])  # Ground truth label

# Ground truth Testing labels        
testing_labels = list(test_labels.values())
testing_labels = np.array(testing_labels)

# Saving testing dataset labels in an excel sheet
import pandas as pd, openpyxl
df = pd.DataFrame(testing_labels)
df.to_excel('Embed check/Testing_labels.xlsx')

# Convert embeddings to NumPy array
test_embeddings = np.array(test_embeddings)

# Saving testing dataset embeddings in an excel sheet
import pandas as pd, openpyxl
df = pd.DataFrame(test_embeddings)
df.to_excel('Embed check/Test_embeddings.xlsx')

# It was observed that the SVC model performed best and achieved highest evaluatio metric scores of Precision, Recall, and F1-score
# The evaluation metric scores reported in the Manuscript were generated using the trained SVC model performance on testing dataset
# Training SVC Machine Learning model
from sklearn import svm

# Define the Support Vector Classifier ML model
clf = svm.SVC(gamma='scale', class_weight='balanced')

# Training the flower classification ML model
# clf.fit(np.array(embedding_list).reshape(-1, 1024), y)
clf.fit(train_emb, y)

# Saving the trained ML model
dump(clf, 'Embed check/svc_model.joblib')

# Predict using the trained model
test_predictions = clf.predict(test_embeddings)
test_predictions = np.array(test_predictions)

# Generate confusion matrix
cm = confusion_matrix(test_y, test_predictions, labels=list(set(test_y)), normalize='true')

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(set(test_y)))
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Generating evaluation metric scores on Testing dataset
print(classification_report(test_y, test_predictions, labels=list(set(test_y))))

# Generating evaluation metric scores
print(precision_score(test_y, test_predictions, average="weighted")) # "macro"
print(recall_score(test_y, test_predictions, average="weighted")) # "macro"
print(f1_score(test_y, test_predictions, average="weighted")) # "macro"
