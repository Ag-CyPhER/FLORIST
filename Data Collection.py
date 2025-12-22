# -*- coding: utf-8 -*-
"""
Created on Tue Jun 3 13:06:07 2025

@author: puran
"""

# FINAL CODE TO DOWNLOAD IMAGES
import pandas as pd
import os
import requests
from openpyxl import load_workbook

# Read the Excel file
df = pd.read_excel('Flower Classification Dataset.xlsx', sheet_name="Training Dataset")
# df = pd.read_excel('Flower Classification Dataset.xlsx', sheet_name="Testing Dataset")

# Initialize variables
current_species = None  # To track the current species
count = 1  # Start count from 1

# Download each image and save it with the specified naming format
for index, row in df.iterrows():
    species = row["Species"]
    gbifID = row["gbifID"]
    url = row["url_ori"]
    FL = row["FL_status"]
    
    # Check if species has changed, reset count if it has
    if species != current_species:
        current_species = species
        count = 1  # Reset the counter for the new species
        
    # Check the category of image
    if FL == "Flower":
        Flow = "F"
    elif FL == "Non Flower":
        Flow = "NF"

    if pd.isna(url) or not isinstance(url, str):
        continue
    
    # Directory folder to store images
    target_folder = "Training"
    # target_folder = "Testing"
    
    # Generate the image filename
    filename = f"{species}_{gbifID}_{Flow}_{count}.jpg"

    if Flow == "F":
        filepath = os.path.join(target_folder+"/Flower", filename)
    else:
        filepath = os.path.join(target_folder+"/Non Flower", filename)
    # filepath = os.path.join(target_folder, filename)

    try:
        # Download the imagem
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for errors

        # Save the image
        with open(filepath, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Downloaded: {filename}")
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")

    count += 1  # Increment the count for the current species
