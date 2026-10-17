import torch

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.utils import get_device


@ModelWrapper.register("torch")
class TorchModelWrapper(ModelWrapper):
    def __init__(self, model_name, model_path=None, config=None, logger=None):
        if config is None:
            config = {}

        super().__init__(model_name=model_name, model_path=model_path, config=config, logger=logger)

        device = config.get('device', None)
        if device is None:
            self._device = get_device()
        else:
            self._device =  torch.device(device)

        self._nfeatures = config.get('nfeatures', 4096)
        self._threshold = config.get('threshold', 0.005)

    def load(self):
        pass
