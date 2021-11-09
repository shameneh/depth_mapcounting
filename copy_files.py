import glob
import shutil
import os
import random
#chose randomly from all images 
images_source = '/home/ubuntu/ameneh/dataset/sliced_all_videos/train/images'
val_output = '/home/ubuntu/ameneh/dataset/sliced_all_videos/validation/images'
midlle_dir ='/home/ubuntu/ameneh/dataset/sliced_all_videos/train/ground_truth_points'
output_path = '/home/ubuntu/ameneh/dataset/sliced_all_videos/validation/ground_truth_points'
if not os.path.isdir(val_output):
    os.mkdir(val_output)
if not os.path.isdir(output_path):
    os.mkdir(output_path)
all_images = glob.glob(images_source+ '/*.jpg')
number_of_sample = int(0.2 * len(all_images))
val_images = random.sample(all_images, number_of_sample)
for val_image in val_images:
    #print(val_images, val_output)
    
    shutil.move(val_image,val_output)
    find_file = os.path.join(midlle_dir,os.path.basename(val_image).replace('jpg','npy'))    
    shutil.move(find_file,output_path)
    #print(find_file,output_path)
