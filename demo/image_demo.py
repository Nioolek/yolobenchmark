# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import mmcv
from mmdet.apis import (inference_detector,
                        init_detector)
from mmyolo.registry import VISUALIZERS
from mmyolo.utils import register_all_modules
from mmyolo.models.utils import model_switch


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', default='dog.jpg', help='Image file')
    # parser.add_argument('--config', default='../configs/yolov5/yolov5_s_16x8_300_coco_v61.py', help='Config file')
    # parser.add_argument('--config', default='../configs/yolov6/yolov6_s_32x8_400_coco.py', help='Config file')
    # parser.add_argument('--config', default='../configs/yolov7/yolov7_l_16x8_300_coco.py', help='Config file')
    parser.add_argument('--config', default='../configs/airdet/airdet_s_300e_coco.py', help='Config file')
    # parser.add_argument('--checkpoint', default='../yolov5s_mm.pt', help='Checkpoint file')
    # parser.add_argument('--checkpoint', default='../yolov6s_mm.pt', help='Checkpoint file')
    # parser.add_argument('--checkpoint', default='../yolov7l_mm.pt', help='Checkpoint file')
    parser.add_argument('--checkpoint', default='../airdet_s_mm.pt', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # register all modules in mmdet into the registries
    register_all_modules()

    # TODO: Support inference of image directory.
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model_switch(model)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # test a single image
    result = inference_detector(model, args.img)

    # show the results
    img = mmcv.imread(args.img)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    visualizer.add_datasample(
        'result',
        img,
        pred_sample=result,
        show=True,
        wait_time=0,
        out_file=args.out_file,
        pred_score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    main(args)
