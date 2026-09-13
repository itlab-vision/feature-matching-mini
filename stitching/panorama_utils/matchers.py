from abc import ABC, abstractmethod
import cv2
import importlib
import numpy as np


class MatcherWrapper(ABC):
    _registry = {}

    def __init__(self, parameters, logger):
        self.params = parameters
        self.logger = logger

    @abstractmethod
    def apply2(self, features):
        pass

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @staticmethod
    def create(matcher_name, parameters, logger):
        matcher_cls = MatcherWrapper._registry.get(matcher_name)
        if matcher_cls is None:
            raise ValueError(f'Matcher {matcher_name} not implemented')
        if logger is not None:
            logger.info(f'Matcher: {matcher_name} ({matcher_cls.__name__})')
        return matcher_cls(parameters, logger)


class OpenCVDetailMatcher(MatcherWrapper):
    def __init__(self, parameters, logger):
        super().__init__(parameters, logger)
        self.matcher = None

    def apply2(self, features):
        pairs = self.matcher.apply2(features)
        self.matcher.collectGarbage()
        return pairs


@MatcherWrapper.register('affine')
class AffineBestOf2NearestMatcher(OpenCVDetailMatcher):
    def __init__(self, parameters, logger):
        super().__init__(parameters, logger)
        self.matcher = cv2.detail_AffineBestOf2NearestMatcher(False,
                                                              self.params['try_cuda'],
                                                              self.params['match_conf'])


@MatcherWrapper.register('2nearest')
class BestOf2NearestMatcher(OpenCVDetailMatcher):
    def __init__(self, parameters, logger):
        super().__init__(parameters, logger)
        self.matcher = cv2.detail_BestOf2NearestMatcher(self.params['try_cuda'],
                                                        self.params['match_conf'])


@MatcherWrapper.register('2nearest_range')
class BestOf2NearestRangeMatcher(OpenCVDetailMatcher):
    def __init__(self, parameters, logger):
        super().__init__(parameters, logger)
        self.matcher = cv2.detail_BestOf2NearestRangeMatcher(self.params['range_width'],
                                                             self.params['try_cuda'],
                                                             self.params['match_conf'])


@MatcherWrapper.register('lightglue')
class LightGlueMatcher(MatcherWrapper):
    def __init__(self, parameters, logger):
        super().__init__(parameters, logger)
        self.torch = importlib.import_module('torch')
        self.device = self.torch.device('cuda' if self.torch.cuda.is_available() else 'cpu')
        self.lightglue = importlib.import_module('lightglue')
        self.model = self.lightglue.LightGlue(
            features=self.params['features']).eval().to(self.device)
        self.match_score_thresh = self.params.get('match_score_thresh', 0.9)

    def prepare_torch_data(self, feat):
        temp_values = []
        temp_scores = []
        for keypoint in feat.keypoints:
            temp_values.append(keypoint.pt)
            temp_scores.append(keypoint.response)
        keypoints = np.array(temp_values)[np.newaxis, ...]
        descriptors = np.copy(feat.descriptors.get()[np.newaxis, ...])
        scores = np.array(temp_scores)[np.newaxis, ...]
        return {'keypoints': self.torch.from_numpy(keypoints).float().to(self.device),
                'descriptors': self.torch.from_numpy(descriptors).float().to(self.device),
                'keypoints_scores': self.torch.from_numpy(scores).float().to(self.device)}

    def _estimate_homography(self, features_left, features_right, matches_info):
        matches_info.H = np.eye(3, dtype=np.float64)
        matches_info.num_inliers = 0
        matches_info.confidence = 0
        cv_matches = matches_info.matches
        if len(cv_matches) < 4:
            return
        try:
            pts1 = np.float32([features_left.keypoints[elem.queryIdx].pt for elem in cv_matches])
            pts2 = np.float32([features_right.keypoints[elem.trainIdx].pt for elem in cv_matches])
            H, inliers_mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
            matches_info.H = H
            matches_info.inliers_mask = inliers_mask.reshape(-1)
            matches_info.num_inliers = int(np.count_nonzero(inliers_mask))
            matches_info.confidence = matches_info.num_inliers / (8.0 + 0.3 * len(matches_info.matches))
        except cv2.error:
            pass

    def apply2(self, features):
        pairwise_matches = [cv2.detail.MatchesInfo()] * len(features) * len(features)
        for i in range(0, len(features)):
            diag = cv2.detail.MatchesInfo()
            diag.src_img_idx = -1
            diag.dst_img_idx = -1
            diag.confidence = 0
            pairwise_matches[i * len(features) + i] = diag
        for i in range(0, len(features)):
            for j in range(i + 1, len(features)):
                feats_left = self.prepare_torch_data(features[i])
                feats_right = self.prepare_torch_data(features[j])
                matches01 = self.model({"image0": feats_left, "image1": feats_right})
                feats_left, feats_right, matches01 = [
                    self.lightglue.utils.rbd(x) for x in [feats_left, feats_right, matches01]
                ]
                matches0, scores0 = matches01["matches0"], matches01['matching_scores0']
                matches_info = cv2.detail.MatchesInfo()
                matches_info.src_img_idx = i
                matches_info.dst_img_idx = j
                cv_matches = []
                for k in range(0, len(matches0)):
                    t = int(matches0[k])
                    if (t >= 0 and t < len(features[j].keypoints)):
                        prob = scores0[k]
                        if (prob >= self.match_score_thresh):
                            cv_matches.append(cv2.DMatch(k, t, float(1.0 - prob.detach().item())))
                matches_info.matches = list(cv_matches)
                self._estimate_homography(features[i], features[j], matches_info)
                pairwise_matches[i * len(features) + j] = matches_info
                matches_info_inv = cv2.detail.MatchesInfo()
                cv_inv_matches = []
                for k in range(0, len(matches0)):
                    t = int(matches0[k])
                    if (t >= 0 and t < len(features[j].keypoints)):
                        prob = scores0[k]
                        if (prob >= self.match_score_thresh):
                            cv_inv_matches.append(cv2.DMatch(t, k, float(1.0 - prob.detach().item())))
                matches_info_inv.src_img_idx = j
                matches_info_inv.dst_img_idx = i
                matches_info_inv.matches = list(cv_inv_matches)
                matches_info_inv.H = np.linalg.inv(matches_info.H)
                matches_info_inv.num_inliers = matches_info.num_inliers
                matches_info_inv.confidence = matches_info.confidence
                pairwise_matches[j * len(features) + i] = matches_info_inv
        return pairwise_matches
