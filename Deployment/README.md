---
title: Practical Deep Learning With Visual Data
emoji: 📚
colorFrom: pink
colorTo: green
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Dress Classifier v1.0
Did you ever wonder what type of dress it is the one you are wearing in that picture? Just upload your .jpg onto this app and find out with this new state-of-the-art dress classifier!
The app supports multiple person detections, meaning that by toggling "Process all detections", you will be able to process images where there are multiple individuals.

Upload a photo with people (preferably wearing dresses), and for each detected person you’ll get:
1) the **original image with boxes**,  
2) the **dress cut-out** (background masked to black),  
3) the **predicted dress type**,  
4) a **Grad-CAM heatmap** showing where the classifier focused at.

## How to use:
1. Drop a `.jpg`/`.png`.  
2. Optional: toggle **“Process all detections”** if there are multiple people.  
3. Click **Run**.

## Dress classification
`casual_dress`, `denim_dress`, `evening_dress`, `jersey_dress`, `knitted_dress`,  
`maxi_dress`, `occasion_dress`, `shift_dress`, `shirt_dress`, `work_dress`.

This implementation runs on the basic CPU