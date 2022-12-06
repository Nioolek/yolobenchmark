import argparse
from collections import OrderedDict
import torch


def convert(src, dst):
    ckpt = torch.load(src, map_location=torch.device('cpu'))
    model = ckpt['model']
    new_state_dict = OrderedDict()
    for k, v in model.items():
        name = k
        if 'head' in name:
            name = name.replace('head', 'bbox_head')
        new_state_dict[name] = v
    data = {'state_dict': new_state_dict}
    torch.save(data, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('--model', default='../../airdet_s_ckpt.pth', help='model name')
    parser.add_argument('--out', default='../../airdet_s_mm.pt', help='save path')
    args = parser.parse_args()

    convert(args.model, args.out)


if __name__ == '__main__':
    main()
