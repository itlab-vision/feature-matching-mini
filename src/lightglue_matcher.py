import torch

from src.dnn_matchers import DNNMatcher
from lightglue import LightGlue
from lightglue.utils import rbd


class LightGlueMatcher(DNNMatcher):
    def __init__(self, logger, matcher_name, descriptor_name, config=None):
        if config is None:
            config = {}

        DNNMatcher.__init__(self, logger, matcher_name, descriptor_name, config)

        if isinstance(descriptor_name, str):
            self._extractor_name = descriptor_name.replace('_lightglue', '').lower()
        else:
            self._extractor_name = descriptor_name._descriptor_name.replace('_lightglue', '').lower()

        self._matcher = LightGlue(features=self._extractor_name, **config).eval().to(self._device)

    def _init_matcher(self):
        pass

    def match(self, features0, features1):
        with torch.no_grad():
            input_dict = {"image0": features0, "image1": features1}
            matches01 = self._matcher(input_dict)
            matches01 = rbd(matches01)
            return {'matches': matches01['matches'], 'scores': matches01['scores']}
