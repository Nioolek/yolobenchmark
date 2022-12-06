from mmyolo.models.layers.yolov6_brick import RepVGGBlock
from mmengine import print_log


def model_switch(model):
    ''' Model switch to deploy status '''
    flag = False
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()
            flag = True
    if flag:
        print_log("Switch model to deploy modality.", 'current')
