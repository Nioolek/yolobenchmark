from mmengine.hooks import Hook
from mmengine.model.wrappers import is_model_wrapper

from mmyolo.registry import HOOKS


@HOOKS.register_module()
class PPYOLOEAssignerHook(Hook):
    priority = 9

    def __init__(self, start_tal_epoch=4):
        super(PPYOLOEAssignerHook, self).__init__()
        self.start_tal_epoch = start_tal_epoch

    def before_train_epoch(self, runner) -> None:
        epochs = runner.epoch
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        if epochs > self.start_tal_epoch:   # TODO > or >=?
            model.bbox_head.use_tal = True
            print('use_tal = True')

