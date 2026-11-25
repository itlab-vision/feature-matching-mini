import os
import sys
import time
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
        self.perf_report = None
        self._stage_times = {}
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
        name = self.parameters.get('features', 'pipeline')
        self.logger.info(f'Create feature extractor: {name}')
        self.features = FeatureExtractor.create(name, self.parameters, self.logger)

    def _get_matcher(self):
        name = self.parameters.get('matcher', 'pipeline')
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

    def _measure_stage(self, stage, t_prev):
        elapsed = time.perf_counter() - t_prev
        self.logger.info(f'Stitch stage time: {stage} = {elapsed:.3f} s')
        self._stage_times[stage] = round(elapsed, 4)
        return time.perf_counter()

    def stitch(self, images):
        t_total = time.perf_counter()
        t_stage = t_total
        self._stage_times = {}
        self.logger.info(f'Stitch: transform images ({self.parameters.get("image_transformer", "opencv")})')
        images_for_features, images_scaling = self.transformer.transform(images)
        t_stage = self._measure_stage('transform images', t_stage)
        self.logger.info(f'Stitch: extract features ({self.parameters.get("features", "pipeline")})')
        features = [self._get_features_from_image(image) for image in images_for_features]
        t_stage = self._measure_stage('extract features', t_stage)
        if hasattr(self.matcher, 'set_images'):
            self.matcher.set_images(images_for_features)
        self.logger.info(f'Stitch: match features ({self.parameters["matcher"]})')
        pairs = self._match_features(features)
        t_stage = self._measure_stage('match features', t_stage)
        images_scaling, images, full_img_sizes = self.subset_selector.select(
            features, pairs, images_scaling, images, self.transformer.full_img_sizes)
        t_stage = self._measure_stage('select subset', t_stage)
        self.logger.info(f'Stitch: estimate homographies ({self.parameters["estimator"]})')
        cameras = self.homography_estimator.estimate(features, pairs)
        t_stage = self._measure_stage('estimate homographies', t_stage)
        self.logger.info(f'Stitch: refine cameras (ba={self.parameters["ba"]})')
        cameras = self.homography_estimator.refine(features, pairs, cameras)
        t_stage = self._measure_stage('refine cameras (ba)', t_stage)
        warped_image_scale = self.wave_corrector.estimate_scale(cameras)
        self.logger.info(f'Stitch: wave correct ({self.parameters["wave_correct"]})')
        cameras = self.wave_corrector.correct(cameras)
        t_stage = self._measure_stage('wave correct', t_stage)
        masks_warped, K = self.image_warper.warp_images(
            images_scaling, cameras, warped_image_scale,
            self.transformer.seam_work_aspect, self.compensator)
        t_stage = self._measure_stage('warp images', t_stage)
        self.logger.info(f'Stitch: compose panorama (blend={self.parameters["blend"]})')
        result = self.panorama_creator.compose(
            images, cameras, masks_warped, warped_image_scale,
            self.transformer.work_scale, full_img_sizes, self.compensator)
        self.perf_report = {
            'stitcher_type': self.parameters.get('stitcher_type'),
            'images_dir': self.parameters.get('images_dir'),
            'out_file_name': self.parameters.get('out_file_name'),
            'num_images': len(images),
            'panorama_resolution': {
                'width': int(result.shape[1]),
                'height': int(result.shape[0]),
            },
            'methods': {
                'detector': self.parameters.get('detector'),
                'descriptor': self.parameters.get('descriptor'),
                'matcher': self.parameters.get('matcher'),
                'estimator': self.parameters.get('estimator'),
                'ba': self.parameters.get('ba'),
                'wave_correct': self.parameters.get('wave_correct'),
                'warp': self.parameters.get('warp'),
                'seam': self.parameters.get('seam'),
                'expos_comp': self.parameters.get('expos_comp'),
                'blend': self.parameters.get('blend'),
            },
            'stage_times_sec': dict(self._stage_times),
            'total_time_sec': round(time.perf_counter() - t_total, 4),
        }
        self.logger.info(f'Stitch: total time = {time.perf_counter() - t_total:.3f} s')
        return result
