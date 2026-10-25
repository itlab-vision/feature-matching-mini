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
    def profile_pipeline(self, detector, descriptor, matcher, img0, img1):
        features0 = detector.detect(img0)
        features0 = descriptor.compute(img0, features0)

        features1 = detector.detect(img1)
        features1 = descriptor.compute(img1, features1)

        res_match = matcher.match(features0, features1)
        return res_match

    @measure_time
    def profile_dnn_extractor(self, extractor, img):
        features = extractor.detectAndCompute(img)
        return features

    @measure_time
    def profile_dnn_pipeline(self, extractor,
                             matcher, img0, img1):
        features0 = extractor.detectAndCompute(img0)
        features1 = extractor.detectAndCompute(img1)

        res_match = matcher.match(features0, features1)
        return res_match

    @measure_time
    def profile_pipeline_methods(self, method, features0, features1):
        result = method.match(features0, features1)
        return result
