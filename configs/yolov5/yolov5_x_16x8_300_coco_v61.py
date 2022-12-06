_base_ = ['./yolov5_s_16x8_300_coco_v61.py']

# model settings
model = dict(
    backbone=dict(depth_multiple=1.33, width_multiple=1.25),
    bbox_head=dict(out_channels=[1.33, 1.25, 1]))
