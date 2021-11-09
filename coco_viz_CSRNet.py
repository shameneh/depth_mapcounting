import argparse
from pathlib import Path
from PIL import Image, ImageDraw
import json
from typing import Tuple
from coco_utils import get_image_infos
from matplotlib import pyplot as plt
import os
import shutil
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('source', type=str, nargs='+')
parser.add_argument('--n', type=int, help='number of images to read')
parser.add_argument('--output', type=str, help='output folder path')
def xywh2xyxy(x, y, w, h) -> Tuple[int]:
    '''convert bbox format from (x, y, w, h) to (x0, y0, x1, y1)'''
    return int(x), int(y), int(x + w), int(y + h)
def center_xy(x,y,w,h) -> Tuple[int]:
    return int(x+(w//2)) , int(y+(h//2))

def main(args):
    for source in args.source:
        
        annotation_path = Path(source, 'annotations/sliced_instances_coco.json')
        #print(Path(source,'labels'))
        #if not os.path.isdir(Path(source,'labels')):
         #   os.mkdir(Path(source,'labels'))
         #   print('is created')
        annotations = json.load(open(annotation_path, 'r'))

        img_infos = get_image_infos(annotations)
        #print(args.n)
        for i, img_info in enumerate(img_infos):
            if args.n and not i < args.n:
                break
           # print(img_info.file_name)
            image_filepath = Path(source, 'sliced_images', img_info.file_name)
            image = Image.open(image_filepath)
            points = []
            for ann_info in img_info.anns:
                # BBOX here!

                if ann_info.category_id == 1:
                    
                    x,y = center_xy(*ann_info.bbox)
                    points.append((x,y))
            frame_num = img_info.file_name
            os.makedirs(Path(args.output,'images'),exist_ok= True)
            os.makedirs(Path(args.output,'ground_truth_points'),exist_ok=True)
            img_path = Path(args.output,'images',Path(source).name + f'_{frame_num}')
            np.save(str(img_path).replace('.jpg','.npy').replace('images','ground_truth_points'), points)
            shutil.copyfile(image_filepath,img_path)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
