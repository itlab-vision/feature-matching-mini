import numpy as np
import cv2 as cv

from src.feature_matcher import FeatureMatcherCV2


class Stitcher:
    _HOMOGRAPHY_METHODS = {
        "ransac": cv.RANSAC,
        "magsac": cv.USAC_MAGSAC,
        "lmeds": cv.LMEDS,
        "rho": cv.RHO
    }

    def __init__(self, logger, detector='sift', descriptor='sift', matcher='bf', config=None):
        if config is None:
            config = {}

        self._homography_threshold = config.pop('homography_threshold', 3.0)
        self._homography_method = config.pop('homography_method', "ransac")
        self._logger = logger
        self._feature_matcher = FeatureMatcherCV2(detector=detector, descriptor=descriptor, matcher=matcher,
                                                  logger=logger, config=config)

    def _is_valid_transformation(self, H):
        if H is None:
            return False

        perspective_strength = np.abs(H[2, 0]) + np.abs(H[2, 1])
        if perspective_strength > 0.15:
            return False

        det = np.linalg.det(H[:2, :2])
        if det < 0.1 or det > 10.0:
            return False

        return True

    def _find_homography(self, keypoints1, keypoints2, matches):
        self._logger.info('START: Find homography / Affine transform')

        if len(matches) < 4:
            self._logger.error('At least 4 matches are required')
            return None

        points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        H, mask = cv.findHomography(points1, points2, self._HOMOGRAPHY_METHODS.get(self._homography_method),
                                    self._homography_threshold)
        if not self._is_valid_transformation(H):
            self._logger.warning('findHomography produced invalid/depth matrix. Trying estimateAffine2D')
            M, mask = cv.estimateAffine2D(points1, points2,
                                          method=self._HOMOGRAPHY_METHODS.get(self._homography_method),
                                          ransacReprojThreshold=self._homography_threshold)
            if M is not None:
                H = np.eye(3, dtype=np.float64)
                H[:2, :] = M
            else:
                self._logger.error('All transformation estimations failed')
                return None

        self._logger.info('FINISH: Find homography')
        return H

    def _check_homography(self, H, img1, img2):
        self._logger.info('START: Check homography')

        if H is None:
            return False

        h1, w1 = img1.shape[:2]
        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)

        try:
            transformed1 = cv.perspectiveTransform(corners1, H)
        except Exception as e:
            self._logger.error(f'Perspective transform error: {e}')
            return False

        transformed1 = transformed1.reshape(-1, 2)

        x = transformed1[:, 0]
        y = transformed1[:, 1]

        transformed_width = x.max() - x.min()
        transformed_height = y.max() - y.min()

        width_ratio = transformed_width / w1
        height_ratio = transformed_height / h1

        self._logger.info(f'Transformed image size: {transformed_width:.1f} x {transformed_height:.1f}')
        self._logger.info(f'Scale ratio: {width_ratio:.2f} x {height_ratio:.2f}')
        self._logger.info('FINISH: Check homography')
        return True

    def _get_features_and_matches(self, img1, img2):
        self._logger.info('START: Feature matching')

        features1, features2, good_matches = self._feature_matcher.match(img1, img2)

        if features1 is None or features2 is None:
            self._logger.error('Feature matcher returned None')
            return None, None, None

        keypoints1 = features1.get('kp') or features1.get('keypoints')
        keypoints2 = features2.get('kp') or features2.get('keypoints')

        if keypoints1 is None or keypoints2 is None:
            self._logger.error('No keypoints detected')
            return None, None, None

        matches = good_matches.get('matches')
        if matches is None:
            self._logger.error('Matches are None')
            return None, None, None

        self._logger.info(f'Good matches: {len(matches)}')
        if len(matches) < 4:
            self._logger.error(f'Not enough matches: {len(matches)}')
            return None, None, None

        self._logger.info('FINISH: Feature matching')
        return keypoints1, keypoints2, matches

    def _match_images(self, img1, img2):
        self._logger.info('START: Match neighbouring images')

        keypoints1, keypoints2, matches = self._get_features_and_matches(img1, img2)

        if keypoints1 is None:
            return None

        H = self._find_homography(keypoints1, keypoints2, matches)
        if H is None:
            return None

        if not self._check_homography(H, img1, img2):
            return None

        self._logger.info('FINISH: Match neighbouring images')
        return H

    def _build_global_homographies(self, pairwise_homographies, num_images):
        self._logger.info('START: Build global homographies')

        ref_idx = num_images // 2
        global_H = [None] * num_images
        global_H[ref_idx] = np.eye(3, dtype=np.float64)

        for i in range(ref_idx - 1, -1, -1):
            H_pair = pairwise_homographies[i]
            global_H[i] = global_H[i + 1] @ H_pair
            global_H[i] = global_H[i] / global_H[i][2, 2]

        for i in range(ref_idx + 1, num_images):
            H_pair_inv = np.linalg.inv(pairwise_homographies[i - 1])
            global_H[i] = global_H[i - 1] @ H_pair_inv
            global_H[i] = global_H[i] / global_H[i][2, 2]

        self._logger.info('FINISH: Build global homographies')
        return global_H

    def _calculate_canvas(self, imgs, global_H):
        self._logger.info('START: Calculate panorama canvas')

        all_corners = []
        for i, img in enumerate(imgs):
            h, w = img.shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            transformed = cv.perspectiveTransform(corners, global_H[i])
            all_corners.append(transformed.reshape(-1, 2))

        all_corners = np.vstack(all_corners)
        x_min = int(np.floor(all_corners[:, 0].min()))
        y_min = int(np.floor(all_corners[:, 1].min()))

        x_max = int(np.ceil(all_corners[:, 0].max()))
        y_max = int(np.ceil(all_corners[:, 1].max()))

        width = x_max - x_min
        height = y_max - y_min

        self._logger.info(f'Canvas: {width} x {height}')
        self._logger.info('FINISH: Calculate panorama canvas')
        return x_min, y_min, width, height

    def _create_panorama(self, imgs, global_H):
        self._logger.info('START: Create panorama')

        x_min, y_min, width, height = self._calculate_canvas(imgs, global_H)
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
        panorama = np.zeros((height, width, 3), dtype=np.uint8)
        panorama_mask = np.zeros((height, width), dtype=np.uint8)

        for i, img in enumerate(imgs):
            self._logger.info(f'Warp image {i + 1}/{len(imgs)}')

            H = translation @ global_H[i]
            warped = cv.warpPerspective(img, H, (width, height))

            mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
            warped_mask = cv.warpPerspective(mask, H, (width, height))
            new_pixels = ((warped_mask > 0) & (panorama_mask == 0))

            panorama[new_pixels] = warped[new_pixels]
            panorama_mask[warped_mask > 0] = 255

        self._logger.info('FINISH: Create panorama')
        return panorama

    def stitch_panorama(self, imgs, H_gt=None):
        self._logger.info('START: Stitch full panorama')

        if imgs is None or len(imgs) < 2:
            self._logger.error('Need at least 2 images')
            return None

        pairwise_homographies = []
        for i in range(len(imgs) - 1):
            self._logger.info(f'PAIR {i} -> {i + 1}')
            if H_gt is not None and i + 1 <= len(H_gt) and H_gt[i] is not None:
                H = H_gt[i]
            else:
                H = self._match_images(imgs[i], imgs[i + 1])

            if H is None:
                self._logger.error(f'Failed for pair {i} -> {i + 1}')
                return None
            pairwise_homographies.append(H)

        global_H = self._build_global_homographies(pairwise_homographies, len(imgs))
        try:
            panorama = self._create_panorama(imgs, global_H)
        except Exception as ex:
            self._logger.error(f'Failed to create panorama: {ex}')
            return None

        return panorama

    def stitch2images(self, img1, img2, H_gt=None):
        return self.stitch_panorama([img1, img2], [H_gt])
