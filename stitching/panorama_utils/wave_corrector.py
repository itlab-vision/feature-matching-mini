from abc import ABC, abstractmethod
import cv2
import numpy as np


class WaveCorrector(ABC):
    _registry = {}

    def __init__(self, parameters, logger=None):
        self.parameters = parameters
        self.logger = logger

    @abstractmethod
    def correct(self, cameras):
        pass

    def estimate_scale(self, cameras):
        focals = sorted(cam.focal for cam in cameras)
        if len(focals) % 2 == 1:
            return focals[len(focals) // 2]
        return (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @staticmethod
    def create(name, parameters, logger=None):
        corrector_cls = WaveCorrector._registry.get(name)
        if corrector_cls is None:
            raise ValueError(f'Wave corrector {name} not implemented')
        return corrector_cls(parameters, logger)


class OpenCVWaveCorrector(WaveCorrector):
    def __init__(self, parameters, wave_correct_type, logger=None):
        super().__init__(parameters, logger)
        self.wave_correct_type = wave_correct_type

    def correct(self, cameras):
        rmats = [np.copy(cam.R) for cam in cameras]
        rmats = cv2.detail.waveCorrect(rmats, self.wave_correct_type)
        for idx, cam in enumerate(cameras):
            cam.R = rmats[idx]
        return cameras


@WaveCorrector.register('horiz')
class HorizWaveCorrector(OpenCVWaveCorrector):
    def __init__(self, parameters, logger=None):
        super().__init__(parameters, cv2.detail.WAVE_CORRECT_HORIZ, logger)


@WaveCorrector.register('vert')
class VertWaveCorrector(OpenCVWaveCorrector):
    def __init__(self, parameters, logger=None):
        super().__init__(parameters, cv2.detail.WAVE_CORRECT_VERT, logger)


@WaveCorrector.register('no')
class NoWaveCorrector(WaveCorrector):
    def correct(self, cameras):
        return cameras
