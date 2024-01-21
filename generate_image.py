import os
import json
import cv2
import numpy as np
from predict import sm, BACKBONE, n_classes, activation, resize_image, preprocess_image, predict_and_visualize
import random

# Model and classes 
#model = # Load model 
classes = ['_background_', 'back_bumper', 'back_glass', 'back_left_door','back_left_light',
               'back_right_door', 'back_right_light', 'front_bumper','front_glass',
               'front_left_door',  'front_left_light', 'front_right_door',  'front_right_light',  
               'hood',  'left_mirror', 'right_mirror', 'tailgate',  'trunk', 'wheel']

# Folder of images
images_folder = 'images/trainingset/' # trainingset - testset

resized_folder = "JPEGImages/trainingset" # trainingset - testset
os.makedirs(resized_folder, exist_ok=True)

# Initialize COCO dict
coco_format = {
    'images': [],
    'categories': [], 
    'annotations': []
}

# Get categories 
for i, cls in enumerate(classes):
    cat = {'id': i+1, 'name': cls}
    coco_format['categories'].append(cat)

# Iterate through images   
id=401 #401 -> trainingset / 902 -> testset
id_annotation=3094 # 3094 -> trainingset / 3920 -> testset
count=1
for i, img_name in enumerate(os.listdir(images_folder)):
    
    # Load and preprocess image
    img_path = os.path.join(images_folder, img_name)
    img = cv2.imread(img_path)
    img_scaled = resize_image(img_path) #preprocess_image(img_path)[0]

    # Save resized image
    # Construct image name
    img_name = "train" + str(id) + ".jpg" # train -> trainingset / te -> testset
    resized_path = os.path.join(resized_folder, img_name)
    cv2.imwrite(resized_path, img_scaled)

    #print(" final : ", final_annotations)
    print("---------------- CURRENTLY : " + str(count) + "----------------------------------" )
    count+=1
   
    id+=1
    id_annotation+=1
# Save annotations            
with open('predictions_trainingset_min_area_1000.json', 'w') as f:
    json_content = json.dumps(coco_format, indent=2)
  
    f.write(json_content)