from sahi.slicing import slice_coco
from sahi.utils.file import load_json
from pathlib import Path
from PIL import Image, ImageDraw
import os
import shutil
import argparse


parser = argparse.ArgumentParser()
camera = False
parser.add_argument('source', type=str, nargs='+')

def main(args):

    for source in args.source:
        annotation_path = Path(source, 'annotations/instances_default.json')
        annotation_dir_path = Path(source, 'annotations')
        images_path = Path(source, 'images')
        output_annotation_file_name ='sliced_instances_coco.json'
        output_dir = Path(source,'sliced_images')

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)


        coco_dict, coco_path = slice_coco(
            coco_annotation_file_path=annotation_path,
            image_dir=images_path,
            output_coco_annotation_file_name='sliced_instances',
            ignore_negative_samples=False,
            output_dir=output_dir,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            min_area_ratio=0.1,
            verbose=True
        )
        shutil.move(Path(output_dir,output_annotation_file_name),Path(annotation_dir_path,output_annotation_file_name))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
