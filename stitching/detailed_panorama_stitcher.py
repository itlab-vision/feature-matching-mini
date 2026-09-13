import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from stitching.panorama_utils.matchers import MatcherWrapper  # noqa: E402
from stitching.panorama_utils.feature_extractor import FeatureExtractor  # noqa: E402
from stitching.panorama_utils.wave_corrector import WaveCorrector  # noqa: E402
from stitching.panorama_utils.homography_estimator import HomographyEstimator  # noqa: E402
from stitching.panorama_utils.image_transformer import ImageTransformer  # noqa: E402
from stitching.panorama_utils.subset_selector import SubsetSelector  # noqa: E402
from stitching.panorama_utils.warper import ImageWarper  # noqa: E402
from stitching.panorama_utils.exposure_compensator import ExposureCompensator  # noqa: E402
from stitching.panorama_utils.panorama_creator import PanoramaCreator  # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stitching.panorama_stitcher import PanoramaStitcher  # noqa: E402


class DetailedPanoramaStitcher(PanoramaStitcher):
    def __init__(self, logger, parameters=None):
        super().__init__(logger, parameters)
        self._get_feature_extractor()
        self._get_matcher()
        self._get_transformer()
        self._get_subset_selector()
        self._get_homography_estimator()
        self._get_wave_corrector()
        self._get_exposure_compensator()
        self._get_image_warper()
        self._get_panorama_creator()

    def _get_feature_extractor(self):
        name = self.parameters['features']
        self.logger.info(f'Create feature extractor: {name}')
        self.features = FeatureExtractor.create(name, self.parameters, self.logger)

    def _get_matcher(self):
        name = self.parameters['matcher']
        self.logger.info(f'Create matcher: {name}')
        self.matcher = MatcherWrapper.create(name, self.parameters, self.logger)

    def _get_transformer(self):
        name = self.parameters.get('image_transformer', 'opencv')
        self.logger.info(f'Create image transformer: {name}')
        self.transformer = ImageTransformer.create(
            name, self.parameters, self.logger)

    def _get_subset_selector(self):
        name = self.parameters.get('subset_selector', 'opencv')
        self.logger.info(f'Create subset selector: {name}')
        self.subset_selector = SubsetSelector.create(
            name, self.parameters, self.logger)

    def _get_homography_estimator(self):
        name = self.parameters['estimator']
        self.logger.info(f'Create homography estimator: {name}')
        self.homography_estimator = HomographyEstimator.create(
            name, self.parameters, self.logger)

    def _get_wave_corrector(self):
        name = self.parameters['wave_correct']
        self.logger.info(f'Create wave corrector: {name}')
        self.wave_corrector = WaveCorrector.create(
            name, self.parameters, self.logger)

    def _get_exposure_compensator(self):
        name = self.parameters['expos_comp']
        self.logger.info(f'Create exposure compensator: {name}')
        self.compensator = ExposureCompensator.create(
            name, self.parameters, self.logger)

    def _get_image_warper(self):
        name = self.parameters.get('image_warper', 'opencv')
        self.logger.info(f'Create image warper: {name}')
        self.image_warper = ImageWarper.create(
            name, self.parameters, self.logger)

    def _get_panorama_creator(self):
        name = self.parameters.get('panorama_creator', 'opencv')
        self.logger.info(f'Create panorama creator: {name}')
        self.panorama_creator = PanoramaCreator.create(
            name, self.parameters, self.logger)

    def _get_features_from_image(self, image):
        return self.features.extract(image)

    def _match_features(self, features_cv):
        return self.matcher.apply2(features_cv)

    def stitch(self, images):
        self.logger.info(f'Stitch: transform images ({self.parameters.get("image_transformer", "opencv")})')
        images_for_features, images_scaling = self.transformer.transform(images)
        self.logger.info(f'Stitch: extract features ({self.parameters["features"]})')
        features = [self._get_features_from_image(image) for image in images_for_features]
        self.logger.info(f'Stitch: match features ({self.parameters["matcher"]})')
        pairs = self._match_features(features)
        images_scaling, images, full_img_sizes = self.subset_selector.select(
            features, pairs, images_scaling, images, self.transformer.full_img_sizes)
        self.logger.info(f'Stitch: estimate homographies ({self.parameters["estimator"]})')
        cameras = self.homography_estimator.estimate(features, pairs)
        self.logger.info(f'Stitch: refine cameras (ba={self.parameters["ba"]})')
        cameras = self.homography_estimator.refine(features, pairs, cameras)
        warped_image_scale = self.wave_corrector.estimate_scale(cameras)
        self.logger.info(f'Stitch: wave correct ({self.parameters["wave_correct"]})')
        cameras = self.wave_corrector.correct(cameras)
        masks_warped, K = self.image_warper.warp_images(
            images_scaling, cameras, warped_image_scale,
            self.transformer.seam_work_aspect, self.compensator)
        self.logger.info(f'Stitch: compose panorama (blend={self.parameters["blend"]})')
        result = self.panorama_creator.compose(
            images, cameras, masks_warped, warped_image_scale,
            self.transformer.work_scale, full_img_sizes, self.compensator)
        return result
