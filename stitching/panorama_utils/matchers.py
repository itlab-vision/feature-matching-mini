from abc import ABC, abstractmethod
import cv2
import numpy as np

from stitching.panorama_utils.utils import (
    SRC_MATCHER_NAMES,
    SRC_PIPELINES,
    image_features_to_src_dict,
    import_src,
    resolve_src_matcher,
    resolve_src_names,
    src_config,
    validate_descriptor_matcher,
)


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


@MatcherWrapper.register('pipeline')
class PipelineMatcher(MatcherWrapper):
    """Matcher backed by the src feature-matching library.

    Usage in config:
        matcher: 'lightglue'   # one of: 'bf', 'flann', 'lightglue',
                               # 'lightglue_opencv', 'superglue', 'loftr', 'roma'
        sequential_matching: true   # optional: match only adjacent pairs
                                    # (0-1, 1-2, ..., n-2-n-1), all other
                                    # pairs get empty entries
        src_config:
          matcher: {mode: 'knn', k: 2}   # optional matcher params

    The generic `matcher: 'pipeline'` form delegates to `src_matcher`
    (e.g. src_matcher: 'flann').

    Dense matchers (loftr, roma) need the source images, provided automatically
    by DetailedPanoramaStitcher via set_images().
    """

    def __init__(self, parameters, logger):
        super().__init__(parameters, logger)
        self.matcher_name = resolve_src_matcher(parameters)
        self.cfg = src_config(parameters)
        self._matcher = None
        self._images = None
        self._canonical = {}
        self._canonical_list = {}

    def set_images(self, images):
        self._images = images

    def _lazy_init(self):
        if self._matcher is not None:
            return
        _, Descriptor, Matcher = import_src()
        self.detector_name, self.descriptor_name = resolve_src_names(self.params)
        validate_descriptor_matcher(self.descriptor_name, self.matcher_name)
        descriptor_cfg = dict(self.cfg.get('descriptor', {}) or {})
        matcher_cfg = dict(self.cfg.get('matcher', {}) or {})
        self.logger.info(f'Src matcher: {self.matcher_name} '
                         f'(descriptor={self.descriptor_name})')
        descriptor_class = Descriptor.create(self.descriptor_name, self.logger, descriptor_cfg)
        self._matcher = Matcher.create(self.matcher_name, self.logger, matcher_cfg, descriptor_class)

    def _to_src_dict(self, features_cv, idx):
        if self.matcher_name in SRC_PIPELINES:
            if self._images is None or idx >= len(self._images):
                self.logger.error(f'No image available for dense matcher pair index {idx}')
                return {'image': None, 'keypoints': (), 'descriptors': ()}
            return {'image': self._images[idx], 'keypoints': (), 'descriptors': ()}
        return image_features_to_src_dict(features_cv)

    def _rescale_keypoints(self, keypoints, idx):
        if self.matcher_name != 'loftr':
            return keypoints
        image = self._images[idx]
        height, width = image.shape[:2]
        scale = min(640.0 / max(height, width), 1.0)
        new_h = (int(height * scale) // 8) * 8
        new_w = (int(width * scale) // 8) * 8
        keypoints = keypoints.copy()
        keypoints[:, 0] *= width / new_w
        keypoints[:, 1] *= height / new_h
        return keypoints

    def _canonical_keypoints(self, idx, keypoints):
        if idx not in self._canonical:
            self._canonical[idx] = keypoints.astype(np.float32, copy=True)
            return np.arange(len(keypoints), dtype=np.int64)
        canon = self._canonical[idx]
        index_map = np.empty(len(keypoints), dtype=np.int64)
        for start in range(0, len(keypoints), 256):
            chunk = keypoints[start:start + 256]
            diff = np.abs(canon[:, None, :] - chunk[None, :, :]).max(axis=2)
            best = diff.argmin(axis=0)
            matched = diff[best, np.arange(len(chunk))] <= 1.0
            for offset, (is_match, best_idx) in enumerate(zip(matched, best)):
                row = start + offset
                if is_match:
                    index_map[row] = int(best_idx)
                else:
                    index_map[row] = len(canon)
                    canon = np.vstack([canon, chunk[offset]])
            self._canonical[idx] = canon
        return index_map

    def _estimate_homography(self, keypoints_left, keypoints_right, matches_info):
        matches_info.H = np.eye(3, dtype=np.float64)
        matches_info.num_inliers = 0
        matches_info.confidence = 0
        cv_matches = matches_info.matches
        if len(cv_matches) < 4:
            return
        try:
            pts1 = keypoints_left
            pts2 = keypoints_right
            H, inliers_mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
            matches_info.H = H
            matches_info.inliers_mask = inliers_mask.reshape(-1)
            matches_info.num_inliers = int(np.count_nonzero(inliers_mask))
            matches_info.confidence = matches_info.num_inliers / (8.0 + 0.3 * len(matches_info.matches))
        except cv2.error:
            pass

    def _match_pair(self, features_left, features_right, out, i, j):
        matches_info = cv2.detail.MatchesInfo()
        matches_info.src_img_idx = i
        matches_info.dst_img_idx = j
        matches_info.matches = []
        matches_info.H = np.eye(3, dtype=np.float64)
        matches_info.num_inliers = 0
        matches_info.confidence = 0
        if out is None or len(out.get('matches', ())) == 0:
            return matches_info
        matches = out['matches']
        if len(matches) > 0 and isinstance(matches[0], (list, tuple)):
            matches = [match[0] for match in matches]
        scores = out.get('scores')
        if self.matcher_name in SRC_PIPELINES:
            keypoints0 = self._rescale_keypoints(np.asarray(out['keypoints0'], dtype=np.float32), i)
            keypoints1 = self._rescale_keypoints(np.asarray(out['keypoints1'], dtype=np.float32), j)
            map0 = self._canonical_keypoints(i, keypoints0)
            map1 = self._canonical_keypoints(j, keypoints1)
            cv_matches = []
            keypoints0_filtered = []
            keypoints1_filtered = []
            for match_idx, match in enumerate(matches):
                score = float(scores[match_idx]) if scores is not None else 1.0
                cv_matches.append(cv2.DMatch(int(map0[int(match.queryIdx)]),
                                             int(map1[int(match.trainIdx)]),
                                             float(1.0 - score)))
                keypoints0_filtered.append(keypoints0[int(match.queryIdx)])
                keypoints1_filtered.append(keypoints1[int(match.trainIdx)])
            matches_info.matches = list(cv_matches)
            self._estimate_homography(np.float32(keypoints0_filtered),
                                      np.float32(keypoints1_filtered), matches_info)
            return matches_info

        cv_matches = list(matches)
        matches_info.matches = cv_matches
        pts1 = []
        pts2 = []
        for match in cv_matches:
            if match.queryIdx < len(features_left.keypoints) and match.trainIdx < len(features_right.keypoints):
                pts1.append(features_left.keypoints[match.queryIdx].pt)
                pts2.append(features_right.keypoints[match.trainIdx].pt)
        if len(pts1) >= 4:
            self._estimate_homography(np.float32(pts1), np.float32(pts2), matches_info)
        return matches_info

    def _store_canonical_keypoints(self, features):
        for idx, canon in self._canonical.items():
            keypoints = tuple(cv2.KeyPoint(float(x), float(y), 1.0, -1, 0.0, 0, k)
                              for k, (x, y) in enumerate(canon))
            features[idx].keypoints = keypoints

    def _iter_pairs(self, n):
        if self.params.get('sequential_matching'):
            return [(i, i + 1) for i in range(n - 1)]
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    @staticmethod
    def _empty_matches_info():
        matches_info = cv2.detail.MatchesInfo()
        matches_info.src_img_idx = -1
        matches_info.dst_img_idx = -1
        matches_info.confidence = 0
        return matches_info

    def apply2(self, features):
        self._lazy_init()
        n = len(features)
        pairwise_matches = [cv2.detail.MatchesInfo()] * n * n
        for i in range(0, n):
            pairwise_matches[i * n + i] = self._empty_matches_info()
        for i, j in self._iter_pairs(n):
            feats_left = self._to_src_dict(features[i], i)
            feats_right = self._to_src_dict(features[j], j)
            out = self._matcher.match(feats_left, feats_right)
            matches_info = self._match_pair(features[i], features[j], out, i, j)
            pairwise_matches[i * n + j] = matches_info
            self.logger.info(f'Pair ({i}, {j}): matches={len(matches_info.matches)}, '
                             f'inliers={matches_info.num_inliers}, confidence={matches_info.confidence:.3f}')

            matches_info_inv = cv2.detail.MatchesInfo()
            matches_info_inv.src_img_idx = j
            matches_info_inv.dst_img_idx = i
            matches_info_inv.matches = [cv2.DMatch(m.trainIdx, m.queryIdx, m.distance)
                                        for m in matches_info.matches]
            matches_info_inv.H = np.linalg.inv(matches_info.H)
            matches_info_inv.num_inliers = matches_info.num_inliers
            matches_info_inv.confidence = matches_info.confidence
            pairwise_matches[j * n + i] = matches_info_inv
        if self.params.get('sequential_matching'):
            computed = set(self._iter_pairs(n))
            for i in range(n):
                for j in range(n):
                    if i != j and (i, j) not in computed and (j, i) not in computed:
                        pairwise_matches[i * n + j] = self._empty_matches_info()
        self._store_canonical_keypoints(features)
        return pairwise_matches


for _src_matcher_name in SRC_MATCHER_NAMES:
    MatcherWrapper.register(_src_matcher_name)(PipelineMatcher)
