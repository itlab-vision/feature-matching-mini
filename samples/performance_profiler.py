import time


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        work_time = end - start
        return result, work_time
    return wrapper


class PerformanceProfiler:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    @measure_time
    def profile_detection(self, detector, img):
        kp = detector.detect(img)
        return kp

    @measure_time
    def profile_descriptor(self, descriptor, img, kp):
        features = descriptor.compute(img, kp)
        return features

    @measure_time
    def profile_matching(self, matcher, features1, features2):
        res_match = matcher.match(features1, features2)
        return res_match

    @measure_time
    def profile_pipeline(self, detector, detector_name, descriptor, descriptor_name,
                         matcher, matcher_name, img0, img1):
        kp0 = detector.detect(img0)
        kp0 = self.preprocessor.prepare_features(kp0, from_algo=detector_name, to_algo=descriptor_name)
        des0 = descriptor.compute(img0, kp0)['des']
        des0 = self.preprocessor.prepare_features(des0, from_algo=descriptor_name, to_algo=matcher_name)

        kp1 = detector.detect(img1)
        kp1 = self.preprocessor.prepare_features(kp1, from_algo=detector_name, to_algo=descriptor_name)
        des1 = descriptor.compute(img1, kp1)['des']
        des1 = self.preprocessor.prepare_features(des1, from_algo=descriptor_name, to_algo=matcher_name)

        res_match = matcher.match({'kp': kp0.get('kp'), 'des': des0, 'img_shape': img0.shape},
                                  {'kp': kp1.get('kp'), 'des': des1, 'img_shape': img1.shape})
        return res_match

    @measure_time
    def profile_dnn_extractor(self, extractor, img):
        features = extractor.detectAndCompute(img)
        return features

    @measure_time
    def profile_dnn_pipeline(self, extractor, extractor_name,
                             matcher, matcher_name, img0, img1):
        features0 = extractor.detectAndCompute(img0)
        features1 = extractor.detectAndCompute(img1)
        features0 = self.preprocessor.prepare_features(features0, from_algo=extractor_name, to_algo=matcher_name)
        features1 = self.preprocessor.prepare_features(features1, from_algo=extractor_name, to_algo=matcher_name)
        res_match = matcher.match(features0, features1)
        return res_match

    @measure_time
    def profile_pipeline_methods(self, method, features0, features1):
        result = method.match(features0, features1)
        return result
