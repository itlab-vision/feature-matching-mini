import torch
from abc import abstractmethod

from src.matchers import Matcher


class DNNMatcher(Matcher, register=False):
    def __init__(self, matcher_name, logger, config=None, descriptor_name=None):
        if config is None:
            config = {}

        Matcher.__init__(self, matcher_name, logger, config, descriptor_name)

        device = config.pop('device', None)
        if device is None:
            if torch.cuda.is_available():
                self._device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self._device = torch.device('mps')
            else:
                self._device = torch.device('cpu')
        else:
            self._device = torch.device(device)

    def _init_matcher(self):
        pass

    @abstractmethod
    def match(self, features0, features1):
        pass
