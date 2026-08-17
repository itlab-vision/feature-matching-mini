import cv2 as cv
from matchers import Matcher


class LightGlueOpenCVMatcher(Matcher):
    def __init__(self, logger, matcher_name, descriptor_name, config):
        super().__init__(logger, matcher_name, descriptor_name)
        self._logger = logger
        if descriptor_name._descriptor_name == 'diskopencv':
            self.lightglue_model_path = config.pop('lightglue_model_path', "models/disk_lightglue_2outputs.onnx")
        else:
            self.lightglue_model_path = config.pop('lightglue_model_path', "models/lightglue_for_aliked.onnx")
        self.scoreThreshold = config.pop('score_threshold', 0.1)
        self.matcher = self._init_matcher()
        self.mode = config.get('mode', 'simple')

    def _init_matcher(self):
        return cv.LightGlueMatcher.create(self.lightglue_model_path, scoreThreshold=self.scoreThreshold)

    def match(self, features1, features2):
        des1 = features1.get('des')
        des2 = features2.get('des')
        kp1 = features1.get('kp')
        kp2 = features2.get('kp')
        img_shape1 = features1.get('img_shape')
        img_shape2 = features2.get('img_shape')
        h1, w1 = img_shape1[:2]
        h2, w2 = img_shape2[:2]

        if not kp1 or not kp2 or des1 is None or des2 is None:
            return {'matches': ()}

        kpts1_mat = cv.KeyPoint_convert(kp1)
        kpts2_mat = cv.KeyPoint_convert(kp2)
        self.matcher.setPairInfo(kpts1_mat, kpts2_mat, (w1, h1), (w2, h2))
        if self.mode == 'knn':
            matches = self.matcher.knnMatch(des1, des2, k=1)
            valid_matches = [m for m in matches if m]
            self._logger.info(f"LightGlue found {len(valid_matches)} matches")
        else:
            matches = self.matcher.match(des1, des2)
            self._logger.info(f"LightGlue found {len(matches)} matches")
        return {'matches': matches}
