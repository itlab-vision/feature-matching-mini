import os
import sys
import cv2
import numpy as np
import importlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stitching.panorama_stitcher import PanoramaStitcher  # noqa: E402


class SuperPointLightGlueStitcher(PanoramaStitcher):
    def __init__(self, logger, parameters=None):
        super().__init__(logger, parameters)
        self.lightglue = importlib.import_module('lightglue')
        self.utils = self.lightglue.utils
        self.torch = importlib.import_module('torch')
        self.device = self.torch.device('cuda' if self.torch.cuda.is_available() else 'cpu')
        self.kp_num = self.parameters['keypoints_num']
        self.descriptor = self.lightglue.SuperPoint(max_num_keypoints=self.kp_num).eval().to(self.device)
        self.matcher = self.lightglue.LightGlue(features='superpoint').eval().to(self.device)

    def _warp_images(self, left_img, right_img, H):
        self.logger.info('START: Warp images')
        height1, width1 = right_img.shape[:2]
        height2, width2 = left_img.shape[:2]

        self.logger.info('Transform corners of the second image')
        points1 = np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)
        points2_ = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)
        points2 = cv2.perspectiveTransform(points2_, H)
        points = np.concatenate((points1, points2), axis=0)

        [x_min, y_min] = np.int32(points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(points.max(axis=0).ravel() + 0.5)

        self.logger.info('Create translation matrix')
        Ht = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]) @ H

        self.logger.info('Create single panorama')
        warped_img = cv2.warpPerspective(left_img, Ht, (x_max - x_min, y_max - y_min))
        warped_img[(-y_min):height1 - y_min, (-x_min):width1 - x_min] = right_img

        self.logger.info('FINISH: Warp images')
        return warped_img

    def _find_homography(self, m_kpts0, m_kpts1):
        self.logger.info('START: Find homography')
        H, _ = cv2.findHomography(m_kpts0.cpu().numpy(), m_kpts1.cpu().numpy(), cv2.RANSAC)
        self.logger.info('FINISH: Find homography')
        return H

    def _numpy_image_to_torch(self, image):
        if image.ndim == 3:
            image = image.transpose((2, 0, 1))
        elif image.ndim == 2:
            image = image[None]
        else:
            raise ValueError(f'Not an image: {image.shape}')
        return self.torch.tensor(image / 255.0, dtype=self.torch.float)

    def _stitch2images(self, img1, img2):
        self.logger.info('START: Stitch two images')
        features1 = self.descriptor.extract(self._numpy_image_to_torch(img1).to(self.device))
        features2 = self.descriptor.extract(self._numpy_image_to_torch(img2).to(self.device))
        m_kpts0, m_kpts1 = self._match_keypoints(features1, features2)
        H = self._find_homography(m_kpts0, m_kpts1)
        output_img = self._warp_images(img1, img2, H)

        self.logger.info('FINISH: Stitch two images')
        return output_img

    def _match_keypoints(self, feats0, feats1):
        self.logger.info('START: Match descriptors')
        matches01 = self.matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [
            self.utils.rbd(x) for x in [feats0, feats1, matches01]
        ]
        kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        self.logger.info('FINISH: Match descriptors')
        return m_kpts0, m_kpts1

    def stitch(self, images):
        self.logger.info('START: Stitch full panorama')
        img1 = images[0]
        for i in range(len(images) - 1):
            img2 = images[i + 1]
            panorama = self._stitch2images(img1, img2)
            img1 = panorama
        self.logger.info('FINISH: Stitch full panorama')
        return img1
