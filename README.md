# FLORIST: Flowering Labeler for Open-source Research-grade Imagery via Self-supervised Transformer 
This repository contains the source code and instruction for running: FLORIST: Flower Labeler for Open-source Research-grade Imagery via Self-supervised Transformer.

## Citizen science observations
Switchgrass, big bluestem, little bluestem, and Indiangrass are four major warm-season perennial grasses of the North American prairie. Research-grade photos of these four species with observation dates before January 1, 2023 were downloaded from the Global Biodiversity Information Facility repository [GBIF](https://www.gbif.org/) using the filters of “Scientific name”, “Present” for Occurrence status, “Human observation” for Basis of record, and “United States of America” for Country or area. A total of 43,861 photos (8,248 of switchgrass [doi: 10.15468/dl.xhrwtk], 11,081 of big bluestem [doi: 10.15468/dl.2j3v6c], 12,465 of Indiangrass [doi: 10.15468/dl.yvurbw], and 12,067 of little bluestem [doi: 10.15468/dl.unst7e]) were obtained. Because multiple images could be taken of the same plant from different angles and/or distances, we defined one location-date combination as one event when evaluating the latitudinal trend.

## FLORIST to classify flowering and non-flowering photos
We developed a computer vision AI named FLORIST to efficiently and effectively screen large numbers of photos taken by citizen scientists to identifying ones with fresh anthers/stigmas. FLORIST consists of the pretrained DINOv2 and the machine learning classifier library. First, FLORIST used the pretrained DINOv2 ViT-L/14 distilled vision transformer (300 million parameters) to extract 1,024 feature embeddings from each photo. Next, FLORIST was trained to classify flowering or non-flowering images based on these 1,024 feature embeddings through a small dataset consisting of 320 images including an equal number of flowering and non-flowering photos from each species. FLORIST evaluated 26 machine learning models included in the [Lazy Predict library](https://lazypredict.readthedocs.io/en/latest/index.html) using a testing set consisting of randomly selected 1,000 photos (manually verified for fresh anthers/stigmas) to identify the best performing model. Three standard metrics for classification (Precision, Recall, and F1-score) were calculated for both categories. As FLORIST performed better in classifying the non-flowering category, after FLORIST filtered out non-flowering photos, we manually verified remaining photos for fresh anthers and/or stigmas.

# FLORIST Workflow
![image](https://github.com/user-attachments/assets/dfe5c489-f717-44b3-81f9-cff16164db6c)

# DinoV2 Installation
Follow the steps below to set up the environment and run the code in this repository.

## Requirements
- Python **>= 3.8**
- PyTorch **>= 1.8**
- Anaconda (recommended)

## Step 1: Create and activate a Conda environment
```bash
conda create -n dinov2_env python=3.8
conda activate dinov2_env
```
## Step 2: Install Python dependencies
Install all required packages using:
```bash
pip install -r requirements.txt
```
## Step 3: Install DINOv2
To extract feature embeddings from the training, testing, and full datasets, you need to install the DINOv2 repository.
Download and install the DINOv2 repository from the link below:
[*DinoV2 Repository*](https://drive.google.com/drive/folders/1cGyOZF1NxCIJlz57usWBLr0NuGJXC2Yg?usp=sharing)

## Step 4: Download the pretrained DINOv2 model
Download the pretrained dinov2_vitl14 model from the link below:
[*dinov2_vitl14*](https://drive.google.com/file/d/1vZKKuPDfi0XXW3IiV78IT34DeiaeokpG/view?usp=sharing)

# Flower Classifiaction 
### Step 1
The Flower classification datasets can be downloaded from the [*Data Collection.py*](https://github.com/Ag-CyPhER/FLORIST/blob/main/Data%20Collection.py) script that reads the Flower classification dataset.xlsx file.

# Train Test
### Step 2
[*Train Test.py*](https://github.com/Ag-CyPhER/FLORIST/blob/main/Train%20Test.py) code extracts feature embeddings from training and testing image datasets using the DINOv2 model (Version: dinov2_vitl14), trains a Support Vector Classifier (SVC) on 320 labeled training images, and evaluates the model's performance on 1000 independent labeled test images. It outputs evaluation metric scores to assess classification accuracy between "Flower" and "Non-Flower" labels as presented in the manuscript.

# Final inference
### Step 3
[*Inference.py*](https://github.com/Ag-CyPhER/FLORIST/blob/main/Inference.py) code extracts the feature embeddings on the images from Full dataset, loads the trained SVC model to predict labels on the full image dataset. It appends these predictions and saves the updated data to a new Excel file.

### References
This work uses the DINOv2 framework from Facebook Research: (https://github.com/facebookresearch/dinov2/tree/main)
