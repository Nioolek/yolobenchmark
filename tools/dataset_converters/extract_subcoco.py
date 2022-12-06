import argparse
import json
import os
import os.path as osp
import shutil
from collections import defaultdict
from pathlib import Path

import mmcv
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

try:
    import panopticapi
except ImportError:
    panopticapi = None


class COCOPanoptic(COCO):
    """This wrapper is for loading the panoptic style annotation file.

    The format is shown in the CocoPanopticDataset class.

    Args:
        annotation_file (str, optional): Path of annotation file.
            Defaults to None.
    """

    def __init__(self, annotation_file = None) -> None:
        super(COCOPanoptic, self).__init__(annotation_file)

    def createIndex(self) -> None:
        """Create index."""
        # create index
        print('creating index...')
        # anns stores 'segment_id -> annotation'
        anns, cats, imgs = {}, {}, {}
        img_to_anns, cat_to_imgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                for seg_ann in ann['segments_info']:
                    # to match with instance.json
                    seg_ann['image_id'] = ann['image_id']
                    img_to_anns[ann['image_id']].append(seg_ann)
                    # segment_id is not unique in coco dataset orz...
                    # annotations from different images but
                    # may have same segment_id
                    if seg_ann['id'] in anns.keys():
                        anns[seg_ann['id']].append(seg_ann)
                    else:
                        anns[seg_ann['id']] = [seg_ann]

            # filter out annotations from other images
            img_to_anns_ = defaultdict(list)
            for k, v in img_to_anns.items():
                img_to_anns_[k] = [x for x in v if x['image_id'] == k]
            img_to_anns = img_to_anns_

        if 'images' in self.dataset:
            for img_info in self.dataset['images']:
                img_info['segm_file'] = img_info['file_name'].replace(
                    'jpg', 'png')
                imgs[img_info['id']] = img_info

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                for seg_ann in ann['segments_info']:
                    cat_to_imgs[seg_ann['category_id']].append(ann['image_id'])

        print('index created!')

        self.anns = anns
        self.imgToAnns = img_to_anns
        self.catToImgs = cat_to_imgs
        self.imgs = imgs
        self.cats = cats


def _process_data(args,
                  type,
                  yolov5_dir='yolov5_coco'):
    convert_yolov5 = args.yolov5
    panoptic = args.panoptic
    year = '2017'

    ann_file_name = f'annotations/instances_{type}{year}.json'
    yolov5_label_name = f'{type}{year}.txt'

    ann_path = osp.join(args.root, ann_file_name)
    json_data = mmcv.load(ann_path)

    new_json_data = {
        'info': json_data['info'],
        'licenses': json_data['licenses'],
        'categories': json_data['categories'],
        'images': [],
        'annotations': []
    }

    images = json_data['images']
    coco = COCO(ann_path)

    if panoptic:
        panoptic_ann_file_name = f'annotations/panoptic_{type}{year}.json'
        ann_path = osp.join(args.root, panoptic_ann_file_name)
        json_data = mmcv.load(ann_path)
        panoptic_new_json_data = {
            'info': json_data['info'],
            'licenses': json_data['licenses'],
            'categories': json_data['categories'],
            'images': [],
            'annotations': []
        }

        panoptic_ann_path = osp.join(args.root, panoptic_ann_file_name)
        coco_panoptic = COCOPanoptic(panoptic_ann_path)

    # shuffle
    np.random.shuffle(images)

    if convert_yolov5:
        mmcv.mkdir_or_exist(osp.join(args.output_dir, yolov5_dir))
        mmcv.mkdir_or_exist(osp.join(args.output_dir, yolov5_dir, 'images'))
        mmcv.mkdir_or_exist(osp.join(args.output_dir, yolov5_dir, 'labels'))
        yolov5_label_name = osp.join(args.output_dir, yolov5_dir,
                                     yolov5_label_name)
        label_file = open(yolov5_label_name, 'w')

    progress_bar = mmcv.ProgressBar(args.num_img)

    for i in range(args.num_img):
        file_name = images[i]['file_name']
        stuff_file_name = osp.splitext(file_name)[0] + '.png'
        image_path = osp.join(args.root, type + year, file_name)
        stuff_image_path = osp.join(args.root, 'stuffthingmaps', type + year,
                                    stuff_file_name)

        ann_ids = coco.getAnnIds(imgIds=[images[i]['id']])
        ann_info = coco.loadAnns(ann_ids)

        new_json_data['images'].append(images[i])
        new_json_data['annotations'].extend(ann_info)

        if panoptic:
            panoptic_stuff_image_path = osp.join(args.root, 'annotations',
                                                 'panoptic_' + type + year,
                                                 stuff_file_name)
            panoptic_ann_info = coco_panoptic.imgToAnns.get(images[i]['id'])
            ann_dict = {
                'segments_info': panoptic_ann_info,
                'file_name': stuff_file_name,
                'image_id': images[i]['id']
            }
            panoptic_new_json_data['images'].append(images[i])
            panoptic_new_json_data['annotations'].append(ann_dict)

            if osp.exists(panoptic_stuff_image_path):
                shutil.copy(
                    panoptic_stuff_image_path,
                    osp.join(args.output_dir, 'annotations',
                             'panoptic_' + type + year, stuff_file_name))

        shutil.copy(image_path, osp.join(args.output_dir, type + year))
        if osp.exists(stuff_image_path):
            shutil.copy(
                stuff_image_path,
                osp.join(args.output_dir, 'stuffthingmaps', type + year))

        image_path = osp.join('./images/' + type + year, file_name)

        if convert_yolov5:
            label_file.write(image_path + '\n')

        progress_bar.update()

    mmcv.dump(new_json_data, osp.join(args.output_dir, ann_file_name))
    if panoptic:
        mmcv.dump(panoptic_new_json_data,
                  osp.join(args.output_dir, panoptic_ann_file_name))

    if convert_yolov5:
        label_file.close()


def _make_dir(output_dir):
    mmcv.mkdir_or_exist(output_dir)
    mmcv.mkdir_or_exist(osp.join(output_dir, 'annotations'))
    mmcv.mkdir_or_exist(osp.join(output_dir, 'annotations/panoptic_train2017'))
    mmcv.mkdir_or_exist(osp.join(output_dir, 'annotations/panoptic_val2017'))
    mmcv.mkdir_or_exist(osp.join(output_dir, 'train2017'))
    mmcv.mkdir_or_exist(osp.join(output_dir, 'val2017'))
    mmcv.mkdir_or_exist(osp.join(output_dir, 'stuffthingmaps/train2017'))
    mmcv.mkdir_or_exist(osp.join(output_dir, 'stuffthingmaps/val2017'))
    mmcv.mkdir_or_exist(osp.join(output_dir, 'stuffthingmaps/val2017'))


def coco91_to_coco80_class(
):  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, None, 24, 25, None, None, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, None, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60, None, None,
        61, None, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, None, 73, 74, 75,
        76, 77, 78, 79, None
    ]
    return x


def convert_to_yolov5(json_file,
                      type,
                      out_dir,
                      use_segments=False,
                      cls91to80=True):
    year = '2017'
    if type == 'train':
        label_out_dir = osp.join(out_dir, 'labels', f'train{year}')
    else:
        label_out_dir = osp.join(out_dir, 'labels', f'val{year}')
    mmcv.mkdir_or_exist(label_out_dir)

    coco80 = coco91_to_coco80_class()

    with open(json_file) as f:
        data = json.load(f)

    # Create image dict
    images = {'%g' % x['id']: x for x in data['images']}

    # Write labels file
    for x in tqdm(data['annotations'], desc=f'Annotations {json_file}'):
        if x['iscrowd']:
            continue

        img = images['%g' % x['image_id']]
        h, w, f = img['height'], img['width'], img['file_name']

        # The COCO box format is [top left x, top left y, width, height]
        box = np.array(x['bbox'], dtype=np.float64)
        box[:2] += box[2:] / 2  # xy top-left corner to center
        box[[0, 2]] /= w  # normalize x
        box[[1, 3]] /= h  # normalize y

        # Segments
        if use_segments:
            segments = [j for i in x['segmentation']
                        for j in i]  # all segments concatenated
            s = (np.array(segments).reshape(-1, 2) /
                 np.array([w, h])).reshape(-1).tolist()

        # Write
        if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
            cls = coco80[x['category_id'] -
                         1] if cls91to80 else x['category_id'] - 1  # class
            line = cls, *(s if use_segments else box)  # cls, box or segments
            with open((Path(label_out_dir) / f).with_suffix('.txt'),
                      'a') as file:
                file.write(('%g ' * len(line)).rstrip() % line + '\n')

    os.symlink(
        osp.join(out_dir, f'../{type}{year}'),
        osp.join(out_dir, f'images/{type}{year}'))


def parse_args():
    parser = argparse.ArgumentParser(description='Extract coco subset')
    parser.add_argument(
        '--root',
        default='/home/PJLAB/huanghaian/dataset/coco',
        help='root path')
    parser.add_argument(
        '--output-dir',
        default='/home/PJLAB/huanghaian/dataset/coco100',
        type=str,
        help='save root dir')
    parser.add_argument(
        '--num-img', default=100, type=int, help='num of extract image')
    parser.add_argument(
        '--need-train',
        action='store_true')
    parser.add_argument(
        '--yolov5',
        action='store_true',
        help='Export to a format that supports yolov5')
    parser.add_argument(
        '--panoptic', action='store_true', help='Process panoptic dataset')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.output_dir != args.root, 'The file will be overwritten in place, so the same folder is not allowed'
    yolov5_dir = 'yolov5_coco'

    _make_dir(args.output_dir)
    if args.need_train:
        print('start processing train dataset')
        _process_data(args, 'train')
    print('start processing val dataset')
    _process_data(args, 'val')

    if args.yolov5:
        print('start processing yolov5')
        if args.need_train:
            convert_to_yolov5(f'{args.output_dir}/annotations/instances_train2017.json',
                              'train',
                              f'{args.output_dir}/{yolov5_dir}')
        convert_to_yolov5(
            f'{args.output_dir}/annotations/instances_val2017.json', 'val',
            f'{args.output_dir}/{yolov5_dir}')
        os.symlink(f'{args.output_dir}/annotations',
                   f'{args.output_dir}/{yolov5_dir}/annotations')


if __name__ == '__main__':
    main()
