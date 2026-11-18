import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stitching.panorama_stitcher import PanoramaStitcher  # noqa: E402


class OpenCVPanoramaStitcher(PanoramaStitcher):
    def __init__(self, logger, parameters=None):
        super().__init__(logger, parameters)

    def stitch(self, images):
        self.logger.info('START: OpenCVPanoramaStitcher.stitch()')
        stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
        status, panorama = stitcher.stitch(images)
        if status != cv2.Stitcher_OK:
            raise Exception(f'Error when stitching images. Error code {status}')
        self.logger.info('FINISH: OpenCVPanoramaStitcher.stitch()')
        return panorama
