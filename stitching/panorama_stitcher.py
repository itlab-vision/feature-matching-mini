from abc import ABC, abstractmethod
import logging


class PanoramaStitcher(ABC):

    def __init__(self, logger, parameters=None):
        self.logger = logger
        self.parameters = {}
        if parameters is not None:
            self.parameters = parameters

    @abstractmethod
    def stitch(self, images):
        pass

    @staticmethod
    def create(logger: logging.Logger, stitcher_type: str = 'OpenCVStitcher',
               parameters: dict = None) -> 'PanoramaStitcher':
        logger.info(f'Create panorama stitcher {stitcher_type}. Parameters: {parameters}')
        if stitcher_type == 'OpenCVStitcher':
            from stitching.opencv_panorama_stitcher import OpenCVPanoramaStitcher
            return OpenCVPanoramaStitcher(logger, parameters)
        elif stitcher_type == 'SuperPointLightGlueStitcher':
            from stitching.superpoint_lightglue_panorama_sticher import SuperPointLightGlueStitcher
            return SuperPointLightGlueStitcher(logger, parameters)
        elif stitcher_type == 'DetailedStitcher':
            from stitching.detailed_panorama_stitcher import DetailedPanoramaStitcher
            return DetailedPanoramaStitcher(logger, parameters)
        else:
            raise ValueError(f'Unsupported stitcher type {stitcher_type}')
