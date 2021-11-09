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
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
import glob

parser = argparse.ArgumentParser()

parser.add_argument('--root', type=str, default='/home/ameneh/Appledataset/data')
parser.add_argument('--train_dir',type=str,help='path to train directory')
parser.add_argument('--val_dir',type=str,help='path to val directory')


def xywh2xyxy(x, y, w, h) -> Tuple[int]:
    '''convert bbox format from (x, y, w, h) to (x0, y0, x1, y1)'''
    return int(x), int(y), int(x + w), int(y + h)


def center_xy(x, y, w, h) -> Tuple[int]:
    return int(x + (w // 2)), int(y + (h // 2))

def save_img_gt(source,annotation_path, out):
    annotations = json.load(open(annotation_path, 'r'))
    img_infos = get_image_infos(annotations)
    # print(args.n)
    for i, img_info in enumerate(img_infos):

        # print(img_info.file_name)
        image_filepath = Path(source, 'images', img_info.file_name)
        image = Image.open(image_filepath)
        points = []
        for ann_info in img_info.anns:
            # BBOX here!

            if ann_info.category_id == 1:
                x, y = center_xy(*ann_info.bbox)
                points.append((x, y))
        frame_num = img_info.file_name
        os.makedirs(Path(out, 'images'), exist_ok=True)
        os.makedirs(Path(out, 'ground_truth_points'), exist_ok=True)
        img_path = Path(out, 'images', Path(source).name + f'_{frame_num}')
        shutil.copyfile(image_filepath, img_path)
        np.save(str(img_path).replace('.PNG', '.npy').replace('images', 'ground_truth_points'), points)
def main():
    for source in data2_sources:
        source_ = Path(ROOT_DIR,source)
        annotation_path = Path(source_, 'annotations/instances_default.json')
        annotations = json.load(open(annotation_path, 'r'))

        tmp_coco = Coco.from_coco_dict_or_path(annotations)

        result = tmp_coco.split_coco_as_train_val(train_split_rate=0.65)

        v_c = result['val_coco'].split_coco_as_train_val(train_split_rate=0.25)


        # export train val split files
        save_json(result["train_coco"].json, os.path.join(source_, 'annotations',"train_split.json"))
        save_json(v_c['val_coco'].json, os.path.join(source_, 'annotations',"val_split.json"))
        save_img_gt(source_, os.path.join(source_, 'annotations',"train_split.json"), TRAIN_SAVE_DIR)
        save_img_gt(source_, os.path.join(source_, 'annotations',"val_split.json"), VAL_SAVE_DIR)



if __name__ == "__main__":
    args = parser.parse_args()
    ROOT_DIR = args.root

    TRAIN_SAVE_DIR = args.train_dir
    VAL_SAVE_DIR = args.val_dir
    data2_sources = []
    f = glob.glob(os.path.join(ROOT_DIR, '*'))
    for i in f:
        if os.path.isdir(i):
            data2_sources.append(os.path.basename(i))
    print(TRAIN_SAVE_DIR, VAL_SAVE_DIR)
    print(data2_sources)
    print(ROOT_DIR)
    
    main()

