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
images_folder = 'images/testset/' # trainingset - testset

resized_folder = "JPEGImages/testset" # trainingset - testset
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
id=1 #839 (new batch 1 ) 2700 (batch 2) -> trainingset /  (old) 960 (new batch)-> testset to validset  / testset -> 1
id_annotation=1 # 4000(old) 6000 (new batch 1) 8000 (batch 2) -> trainingset / 4000(old) 5000 (new batch) (new batch)-> testset to validset  / testset -> 1
count=1
print("length image folder : ",len(os.listdir(images_folder)))
for i, img_name in enumerate(os.listdir(images_folder)):
    
    
    # Load and preprocess image
    img_path = os.path.join(images_folder, img_name)
    img = cv2.imread(img_path)
    img_scaled = resize_image(img_path) #preprocess_image(img_path)[0]

    # Save resized image
    # Construct image name
    img_name = "te" + str(id) + ".jpg" # train -> trainingset / te -> testset
    resized_path = os.path.join(resized_folder, img_name)
    cv2.imwrite(resized_path, img_scaled)
    
    # Add image info
    #'''
    h, w = img_scaled.shape[:2] #, _ = img_scaled.shape
    #print("img_scaledclear : ",img_scaled.shape, " h : ",h, " w : ",w)
    image = {
        'id': id,
        'dataset_id': 1,
        'category_ids': [],
        'path': resized_path, #f'JPEGImages/{img_name}'  
        'width': w,
        'height': h,
        'file_name': img_name,
        'annotated': False,
        'annotating': [],
        'num_annotations': 0, 
        'metadata': {},
        'deleted': False,
        'milliseconds': 0,
        'events': [],
        'regenerate_thumbnail': False
    }
    
    coco_format['images'].append(image)

    # Initialize the model
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    
    # Load model weights
    model.load_weights('best_model_hugging.h5')


    #'''
    # Run prediction
    annotations = predict_and_visualize(model, resized_path)
    #print("annotations : ",annotations)
    #print("annotations : ",annotations)
    

    ''' 
    last_annotation = annotations[-1]
    segmentation = last_annotation['segmentation'][0]
    print("segmentation : ", segmentation)
    '''

    #'''
    final_annotations=[]
    from PIL import Image
    from PIL import ImageDraw
    idx=0
    for annotation in annotations :
        #segmentation = annotation['segmentation'][0]
        if float(annotation['area']) < 1000 :
            #print(" idx : ",idx)
            continue
    

        '''
        #---------- mask -> uselesss ----------------
        # Get segmentation coords 
        seg = annotation['segmentation'][0]

        # Create empty mask
        mask = np.zeros((512, 512), dtype=np.uint8) 

        # Draw polygon on mask 
        cv2.fillPoly(mask, [np.array(seg).reshape((-1, 2))], 1)

        # Now resize mask
        mask = cv2.resize(mask, (512, 512))
        # Update segmentation 
        annotation['segmentation'] = [mask.tolist()]
        '''
        
        #print("annotation['segmentation'] : ", annotation['segmentation'])
        
        '''
        #---------- visualize image ----------------
        img = Image.open(resized_path).convert('RGB')
        
        draw = ImageDraw.Draw(img)
        # Reshape segmentation to tuples
        seg_tuples = []
        for i in range(0, len(segmentation)-1, 2):
            x = segmentation[i] 
            y = segmentation[i+1]
            seg_tuples.append((x, y))

        # Get last vertex 
        #x = segmentation[-1]  
        #seg_tuples.append((x, 0))
        print("seg_tuples : ", seg_tuples)
        print("len seg_tuples : ", len(seg_tuples))
        
        if len(seg_tuples) < 2 :
            continue
        
        # Draw polygon
        draw.polygon(seg_tuples, outline=(0,255,0), width=5)
        annotated_img_path = str(idx)+'_image_with_contours.jpg'
        img.save(annotated_img_path)
        idx+=1
        #'''

        #'''
        #---------- write into coco ----------------
        final_annotations.append(annotation)
        rgb = random.choices(range(256), k=3)
        hex_color = f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'

        data_annotation = {
                'id': id_annotation,
                'image_id': id,
                'category_id': annotation['category_id'],
                'segmentation': annotation['segmentation'],
                'area': annotation['area'],
                'bbox': annotation['bbox'],
                "iscrowd": False,
                "isbbox": False,
                "color": hex_color,
                "metadata": {}
            }
        coco_format['annotations'].append(data_annotation)
        id_annotation+=1
    







    #print(" final : ", final_annotations)
    print("---------------- CURRENTLY : " + str(count) + "----------------------------------" )
    count+=1
   
    id+=1
    id_annotation+=1
# Save annotations            
with open('predictions_testset_min_area_1000.json', 'w') as f:
    json_content = json.dumps(coco_format, indent=2)
  
    f.write(json_content)