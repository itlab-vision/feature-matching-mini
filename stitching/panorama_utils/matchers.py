from abc import ABC, abstractmethod
import cv2
import numpy as np

from src.algorithms import ALL_MATCHERS, DNN_PIPELINES

from stitching.panorama_utils.utils import (
    image_features_to_src_dict,
    import_src,
    resolve_src_matcher,
    resolve_src_names,
    src_stage_config,
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
            if resolve_src_matcher({'matcher': matcher_name}) not in ALL_MATCHERS:
                raise ValueError(f'Matcher {matcher_name} not implemented')
            matcher_cls = PipelineMatcher
        if logger is not None:
            logger.info(f'Matcher: {matcher_name} ({matcher_cls.__name__})')
        return matcher_cls({**parameters, 'matcher': matcher_name}, logger)


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
        self._matcher = None
        self._images = None
        self._source_features = None
        self._descriptor = None
        self._canonical = {}

    def set_images(self, images, source_features=None, descriptor=None):
        self._images = images
        self._source_features = source_features
        self._descriptor = descriptor

    def _lazy_init(self):
        if self._matcher is not None:
            return
        _, self.descriptor_name = resolve_src_names(self.params)
        descriptor_cfg = src_stage_config(self.params, 'descriptor', self.descriptor_name)
        matcher_cfg = self._matcher_config()
        descriptor_backend = descriptor_cfg['backend']
        matcher_backend = matcher_cfg['backend']
        validate_descriptor_matcher(self.descriptor_name, self.matcher_name,
                                    descriptor_backend, matcher_backend)
        _, Descriptor, Matcher = import_src(descriptor_backend, matcher_backend)
        descriptor = self._descriptor
        if descriptor is None and self.matcher_name not in DNN_PIPELINES:
            descriptor = Descriptor.create(self.descriptor_name, self.logger, descriptor_cfg)
        self._matcher = Matcher.create(self.matcher_name, self.logger, matcher_cfg, descriptor)

    def _matcher_config(self):
        return src_stage_config(self.params, 'matcher', self.matcher_name)

    def _match_features(self, left, right):
        return self._matcher.match(left, right)

    def _to_src_dict(self, features_cv, idx):
        if self.matcher_name in DNN_PIPELINES:
            if self._images is None:
                raise ValueError(f'Dense matcher {self.matcher_name!r} requires the input images')
            return {'image': self._images[idx]}
        if self._source_features is not None:
            return self._source_features[idx]
        features = image_features_to_src_dict(features_cv)
        features['extractor_name'] = self.descriptor_name
        return features

    @staticmethod
    def _keypoint_coordinates(keypoints):
        if len(keypoints) and isinstance(keypoints[0], cv2.KeyPoint):
            return cv2.KeyPoint_convert(keypoints)
        return np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)

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

    @staticmethod
    def _center_points(features, points):
        # OpenCV's panorama estimator expects homographies relative to the image centers.
        return points - np.asarray(features.img_size, dtype=np.float32) / 2

    @staticmethod
    def _store_geometry(matches_info, transform, inliers):
        if transform is None or np.linalg.matrix_rank(transform) < 3:
            return
        matches_info.H = transform
        matches_info.inliers_mask = inliers.ravel()
        matches_info.num_inliers = int(np.count_nonzero(inliers))
        matches_info.confidence = matches_info.num_inliers / (8.0 + 0.3 * len(matches_info.matches))

    def _estimate_geometry(self, features_left, features_right, points_left, points_right, matches_info):
        if len(points_left) < 4:
            return
        left = self._center_points(features_left, points_left)
        right = self._center_points(features_right, points_right)
        homography, inliers = cv2.findHomography(left, right, cv2.RANSAC, 3.0)
        self._store_geometry(matches_info, homography, inliers)

    def _match_pair(self, features_left, features_right, out, i, j):
        matches_info = cv2.detail.MatchesInfo()
        matches_info.src_img_idx = i
        matches_info.dst_img_idx = j
        matches = out.get('matches') if out else None
        if matches is None or len(matches) == 0:
            return matches_info
        if isinstance(matches[0], (list, tuple)):
            matches = [row[0] for row in matches if row]
        if self.matcher_name in DNN_PIPELINES:
            keypoints0 = self._keypoint_coordinates(out['keypoints0'])
            keypoints1 = self._keypoint_coordinates(out['keypoints1'])
            map0 = self._canonical_keypoints(i, keypoints0)
            map1 = self._canonical_keypoints(j, keypoints1)
            matches_info.matches = [
                cv2.DMatch(int(map0[m.queryIdx]), int(map1[m.trainIdx]), m.distance) for m in matches
            ]
        else:
            keypoints0 = self._keypoint_coordinates(features_left.keypoints)
            keypoints1 = self._keypoint_coordinates(features_right.keypoints)
            matches = [m for m in matches if 0 <= m.queryIdx < len(keypoints0) and 0 <= m.trainIdx < len(keypoints1)]
            matches_info.matches = list(matches)

        matches_info.inliers_mask = np.zeros(len(matches), dtype=np.uint8)
        self._estimate_geometry(
            features_left, features_right,
            keypoints0[[m.queryIdx for m in matches]],
            keypoints1[[m.trainIdx for m in matches]], matches_info,
        )
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

    def apply2(self, features):
        self._lazy_init()
        self._canonical.clear()
        n = len(features)
        pairwise_matches = [cv2.detail.MatchesInfo() for _ in range(n * n)]
        for i, j in self._iter_pairs(n):
            feats_left = self._to_src_dict(features[i], i)
            feats_right = self._to_src_dict(features[j], j)
            if self.matcher_name not in DNN_PIPELINES and not (features[i].keypoints and features[j].keypoints):
                continue
            out = self._match_features(feats_left, feats_right)
            matches_info = self._match_pair(features[i], features[j], out, i, j)
            pairwise_matches[i * n + j] = matches_info
            self.logger.info(f'Pair ({i}, {j}): matches={len(matches_info.matches)}, '
                             f'inliers={matches_info.num_inliers}, confidence={matches_info.confidence:.3f}')

            matches_info_inv = cv2.detail.MatchesInfo()
            matches_info_inv.src_img_idx = j
            matches_info_inv.dst_img_idx = i
            matches_info_inv.matches = [cv2.DMatch(m.trainIdx, m.queryIdx, m.distance)
                                        for m in matches_info.matches]
            if matches_info.H is not None:
                matches_info_inv.H = np.linalg.inv(matches_info.H)
            if matches_info.matches:
                matches_info_inv.inliers_mask = matches_info.inliers_mask
            matches_info_inv.num_inliers = matches_info.num_inliers
            matches_info_inv.confidence = matches_info.confidence
            pairwise_matches[j * n + i] = matches_info_inv
        self._store_canonical_keypoints(features)
        return pairwise_matches


@MatcherWrapper.register('2nearest')
@MatcherWrapper.register('2nearest_range')
@MatcherWrapper.register('affine')
class TwoNearestMatcher(PipelineMatcher):
    """OpenCV panorama matching policies using the src FLANN matcher."""

    def __init__(self, parameters, logger):
        super().__init__(parameters, logger)
        self.mode = self.matcher_name
        self.matcher_name = 'flann'
        self.distance_ratio = 1.0 - parameters['match_conf']

    def _matcher_config(self):
        return {'backend': 'opencv', 'mode': 'knn', 'k': 2}

    def _iter_pairs(self, n):
        width = self.params['range_width'] if self.mode == '2nearest_range' else n
        return ((i, j) for i in range(n) for j in range(i + 1, min(n, i + width + 1)))

    def _ratio_matches(self, query, train):
        if query['descriptors'] is None or train['descriptors'] is None:
            return []
        if not len(query['descriptors']) or len(train['descriptors']) < 2:
            return []
        neighbors = self._matcher.match(query, train)['matches']
        return [row[0] for row in neighbors
                if len(row) >= 2 and row[0].distance < self.distance_ratio * row[1].distance]

    def _match_features(self, left, right):
        matches = self._ratio_matches(left, right)
        seen = {(m.queryIdx, m.trainIdx) for m in matches}
        for match in self._ratio_matches(right, left):
            if (match.trainIdx, match.queryIdx) not in seen:
                matches.append(cv2.DMatch(match.trainIdx, match.queryIdx, match.distance))
        return {'matches': matches}

    def _estimate_geometry(self, features_left, features_right, points_left, points_right, matches_info):
        if len(points_left) < 6:
            return
        if self.mode == 'affine':
            affine, inliers = cv2.estimateAffinePartial2D(points_left, points_right)
            if affine is not None:
                self._store_geometry(matches_info, np.vstack((affine, [0, 0, 1])), inliers)
            return

        left = self._center_points(features_left, points_left)
        right = self._center_points(features_right, points_right)
        homography, inliers = cv2.findHomography(left, right, cv2.RANSAC, 3.0)
        self._store_geometry(matches_info, homography, inliers)
        if matches_info.confidence > 3.0:
            matches_info.confidence = 0.0
        if matches_info.num_inliers >= 6:
            mask = inliers.ravel().astype(bool)
            refined, _ = cv2.findHomography(left[mask], right[mask], cv2.RANSAC, 3.0)
            if refined is not None and np.linalg.matrix_rank(refined) == 3:
                matches_info.H = refined
