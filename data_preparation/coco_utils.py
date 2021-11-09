from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime
import json


@dataclass
class AnnotationInfo:
    segmentation: List = field(default_factory=list)
    area: float = 0
    iscrowd: int = 0
    image_id: int = 0
    bbox: List = field(default_factory=list)
    category_id: int = 0
    id: int = 0
    attributes: Dict = field(default_factory=dict)


@dataclass
class ImageInfo:
    license: int = 0
    file_name: str = ''
    coco_url: str = ''
    height: int = 0
    width: int = 0
    date_captured: str = ''
    flickr_url: str = ''
    id: int = 0
    anns: List[AnnotationInfo] = field(default_factory=list)


@dataclass
class CategoryInfo:
    supercategory: str = ''
    id: int = 0
    name: str = ''


def get_image_infos(annotations) -> List[ImageInfo]:
    '''get image information from COCO annotations'''
    img_infos = {}

    for img_info in annotations['images']:
        img_info = ImageInfo(**img_info)
        img_infos[img_info.id] = img_info

    for ann_info in annotations['annotations']:
        ann_info = AnnotationInfo(**ann_info)
        img_infos[ann_info.image_id].anns.append(ann_info)

    return list(img_infos.values())
