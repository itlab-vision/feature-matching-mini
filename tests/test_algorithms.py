from src.algorithms import (BACKEND_ALGORITHMS, OPENCV_DETECTORS, OPENCV_DESCRIPTORS, OPENCV_MATCHERS,
                            get_compatible_descriptors,
                            get_compatible_matchers, get_default_backend)


class TestBackendAlgorithms:
    def test_contains_all_supported_backends(self):
        assert set(BACKEND_ALGORITHMS) == {'opencv', 'torch', 'onnx', 'executorch'}

    def test_resolves_default_backend(self):
        assert get_default_backend('sift', 0) == 'opencv'
        assert get_default_backend('superpoint', 0) == 'torch'
        assert get_default_backend('tfeat', 1) == 'torch'
        assert get_default_backend('bf', 2) == 'opencv'

    def test_opencv_model_names_do_not_have_backend_suffix(self):
        assert {'disk', 'aliked'} <= OPENCV_DETECTORS
        assert {'disk', 'aliked'} <= OPENCV_DESCRIPTORS
        assert 'lightglue' in OPENCV_MATCHERS
        assert not any(name.endswith('_opencv') for name in OPENCV_DETECTORS | OPENCV_DESCRIPTORS | OPENCV_MATCHERS)


class TestDetectorDescriptorCompatibility:
    def test_allows_opencv_detector_with_executorch_patch_descriptor(self):
        descriptors = get_compatible_descriptors('sift', 'opencv', 'executorch')
        assert descriptors == ['tfeat', 'hardnet']

    def test_requires_same_backend_for_joint_feature_extractor(self):
        assert get_compatible_descriptors(
            'superpoint_lightglue', 'executorch', 'torch'
        ) == []
        assert get_compatible_descriptors(
            'superpoint_lightglue', 'executorch', 'executorch'
        ) == ['superpoint_lightglue']


class TestDescriptorMatcherCompatibility:
    def test_rejects_executorch_features_for_torch_lightglue(self):
        matchers = get_compatible_matchers(
            'superpoint_lightglue', 'executorch', 'torch'
        )
        assert matchers == []

    def test_accepts_executorch_features_for_executorch_lightglue(self):
        matchers = get_compatible_matchers(
            'superpoint_lightglue', 'executorch', 'executorch'
        )
        assert matchers == ['lightglue']

    def test_accepts_executorch_patch_descriptor_with_opencv_matcher(self):
        matchers = get_compatible_matchers('tfeat', 'executorch', 'opencv')
        assert matchers == ['bf', 'flann']
