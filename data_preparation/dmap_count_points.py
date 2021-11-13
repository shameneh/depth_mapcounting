'''
This script is for generating the ground truth density map 
'''
import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm
import sys
import csv
sys.path.insert(1, '..')
from config import Config
def generate_fixed_kernel_densitymap(image,points,sigma=15):
    '''
    Use fixed size kernel to construct the ground truth density map 
    for ShanghaiTech PartB. 
    image: the image with type numpy.ndarray and [height,width,channel]. 
    points: the points corresponding to heads with order [col,row]. 
    sigma: the sigma of gaussian_kernel to simulate a head. 
    '''
    # the height and width of the image
    image_h = image.shape[0]
    image_w = image.shape[1]

    # coordinate of heads in the image
    points_coordinate = points
    # quantity of heads in the image
    points_quantity = len(points_coordinate)

    # generate ground truth density map
    densitymap = np.zeros((image_h, image_w))
#    print(points_quantity)
    for point in points_coordinate:
        c = min(int(round(point[0])),image_w-1)
        r = min(int(round(point[1])),image_h-1)
        # point2density = np.zeros((image_h, image_w), dtype=np.float32)
        # point2density[r,c] = 1
        densitymap[r,c] = 1
    # densitymap += gaussian_filter(point2density, sigma=sigma, mode='constant')
    densitymap = gaussian_filter(densitymap, sigma=sigma,mode='constant')
    if points_quantity >0:
        densitymap = densitymap / densitymap.sum() * points_quantity
    return densitymap    
    
if __name__ == '__main__':
    cfg = Config()
    phase_list =['train','validation']
    root_dir = '../../dataset/all_data'
    out_path = '.'
    from pathlib import Path
    new_size =cfg.image_size # None
    for phase in phase_list:
        header = ['image','gt_count']


        dir_csv = Path(out_path , 'count_'+phase+'.csv')
        with open(dir_csv, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        
        image_file_list = os.listdir(os.path.join(root_dir,phase,'images'))
        for image_file in tqdm(image_file_list):
            image_path = os.path.join(root_dir, phase, 'images', image_file)
            points_path = image_path.replace('images','ground_truth_points').replace('.PNG','.npy')
            points = np.load(points_path)
            data_csv = [image_file,str(len(points))]
            with open(dir_csv, 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(data_csv)
