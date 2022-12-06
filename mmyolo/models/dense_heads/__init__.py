from .yolov5_head import YOLOV5Head
from .yolov6_head import YOLOV6Head
from .yolov7_head import YOLOV7Head
from .airdet_head import GFocalHead_Tiny
from .ppyoloe_head import PPYOLOEHead

__all__ = ['YOLOV5Head', 'YOLOV6Head', 'YOLOV7Head', 'GFocalHead_Tiny',
           'PPYOLOEHead']
