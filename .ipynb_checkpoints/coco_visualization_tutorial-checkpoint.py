%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import json, cv2
from PIL import Image
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# load anno using file
with open("coco.json", 'r') as f:
    content = f.readlines()
    
coco_json = json.loads(content[0])


idx = 0 # image number
file_name = coco_json['images'][idx]['file_name']
image_id = coco_json['images'][idx]['id']
print(file_name)
print(image_id)

# Mask
image = np.array(Image.open(file_name))
print(image.shape)

def annToMask(anno, height, width):
    arr = np.array(anno['segmentation'][0]).reshape(-1, 2) # contour format (num_pts, 1, 2)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask = cv2.drawContours(image=mask, contours=[arr.astype(np.int32)[:, np.newaxis,:]], contourIdx=0, color=1, thickness=cv2.FILLED)
    return mask
    
    
#### GENERATE and overlay the BINARY MASKS ####
anns = coco_json['annotations']
mask = np.zeros(image.shape[:2])
for i in range(len(anns)):
    if anns[i]['image_id']==image_id:
        mask_temp = annToMask(anns[i], image.shape[0], image.shape[1])
        mask = np.maximum(mask_temp*(i+1), mask)
plt.imshow(mask)


# random colors
from itertools import permutations
import random
colors = [x for x in permutations([255, 128, 64, 0], 3)] # generate colors using combination C(5, 3) = 5!/(5-3)! = 5*4*3 = 60
random.shuffle(colors)
print("number of colors:", len(colors))


mask_new = np.zeros_like(image)
mask_new[:,:, 3] = 255
idx_list = [int(x) for x in np.unique(mask)]
for iidx, i in enumerate(idx_list):
    if i>0: # skip background
        mask_new[mask==i, 0] = colors[iidx][0]
        mask_new[mask==i, 1] = colors[iidx][1]
        mask_new[mask==i, 2] = colors[iidx][2]
        
from PIL import Image
Image.fromarray(mask_new)

#### Weighted original image
alpha = 0.3
masked_image = cv2.addWeighted(image, 1 - alpha, mask_new, alpha, 0)
Image.fromarray(masked_image)# .save('1-1.png')


# Draw Contour

image_cnt = np.array(Image.open(file_name))
for iidx, i in enumerate(idx_list): 
    if i>0: # skip background
        ret, thresh = cv2.threshold((mask==i).astype(np.uint8), 0.5, 255, 0)
        contours, _ = cv2.findContours((mask==i).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for j in range(len(contours)):
            image_cnt = cv2.drawContours(image_cnt, contours, j, (colors[iidx][0],colors[iidx][1],colors[iidx][2],255), 3)
            
Image.fromarray(image_cnt)# .save("1.png")