import argparse
import os
import subprocess
import urllib
from collections import OrderedDict
from pathlib import Path

import requests
import torch


def safe_download(file, url, url2=None, min_bytes=1E0, error_msg=''):
    # Attempts to download file from url or url2, checks and removes incomplete downloads < min_bytes
    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        print(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file))
        assert file.exists(
        ) and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        file.unlink(missing_ok=True)  # remove partial downloads
        print(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -"
                  )  # curl download, retry and resume on fail
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            file.unlink(missing_ok=True)  # remove partial downloads
            print(f'ERROR: {assert_msg}\n{error_msg}')
        print('')


def attempt_download(file,
                     repo='WongKinYiu/yolov7'
                     ):  # from utils.downloads import *; attempt_download()
    # Attempt file download if does not exist
    file = Path(str(file).strip().replace("'", ''))

    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(
            str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(('http:/', 'https:/')):  # download
            url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
            file = name.split('?')[
                0]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                print(f'Found {url} locally at {file}')  # file already exists
            else:
                safe_download(file=file, url=url, min_bytes=1E5)
            return file

        # GitHub assets
        file.parent.mkdir(
            parents=True, exist_ok=True)  # make parent dir (if required)
        try:
            response = requests.get(
                f'https://api.github.com/repos/{repo}/releases/latest').json(
            )  # github api
            assets = [
                x['name'] for x in response['assets']
            ]  # release assets, i.e. ['yolov5s.pt', 'yolov5m.pt', ...]
            tag = response['tag_name']  # i.e. 'v1.0'
        except Exception:  # fallback plan
            assets = [
                'yolov7.pt'
            ]
            try:
                tag = subprocess.check_output(
                    'git tag', shell=True,
                    stderr=subprocess.STDOUT).decode().split()[-1]
            except Exception:
                tag = 'v0.1'  # current release

        if name in assets:
            safe_download(
                file,
                url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
                # url2=f'https://storage.googleapis.com/{repo}/ckpt/{name}',  # backup url (optional)
                min_bytes=1E5,
                error_msg=
                f'{file} missing, try downloading from https://github.com/{repo}/releases/'
            )

    return str(file)


def convert(src, dst):
    ckpt = torch.load(src, map_location=torch.device('cpu'))
    # 保存的模型是重参数前的模型
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float()  # 没有存储 ema 模型
    new_state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        name = k
        if 'model' in k:
            if 'model.105.m' in k and 'implicit' not in k:
                name = k.replace('model.105.m', 'bbox_head.m')
            else:
                name = k.replace('model', 'bbox_head.det')
        if k.find('anchors') >= 0 or k.find('anchor_grid') >= 0:
            continue

        new_state_dict[name] = v
    data = {'state_dict': new_state_dict}
    torch.save(data, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('--model', default='yolov7.pt', help='model name')
    parser.add_argument('--out', default='yolov7l_mm.pt', help='save path')
    args = parser.parse_args()

    assets = [
        'yolov7.pt'
    ]
    assert args.model in assets

    attempt_download(args.model)
    convert(args.model, args.out)


##################################################################
# 由于 pickle 原因，本程序无法直接运行！！！
# 该程序必须放在yolov7工程根目录下，读取原始权重后，会自动转换为 mm 支持的模型
##################################################################
if __name__ == '__main__':
    main()
