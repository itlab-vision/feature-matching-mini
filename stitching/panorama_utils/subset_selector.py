from abc import ABC, abstractmethod
import cv2


class SubsetSelector(ABC):
    _registry = {}

    def __init__(self, parameters, logger=None):
        self.parameters = parameters
        self.logger = logger

    @abstractmethod
    def select(self, features, pairs, scaled_imgs, full_images, full_img_sizes):
        pass

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @staticmethod
    def create(name, parameters, logger=None):
        selector_cls = SubsetSelector._registry.get(name)
        if selector_cls is None:
            raise ValueError(f'Subset selector {name} not implemented')
        return selector_cls(parameters, logger)


@SubsetSelector.register('opencv')
class OpenCVSubsetSelector(SubsetSelector):
    def __init__(self, parameters, logger=None):
        super().__init__(parameters, logger)
        self.conf_thresh = self.parameters['conf_thresh']

    def select(self, features, pairs, scaled_imgs, full_images, full_img_sizes):
        indices = cv2.detail.leaveBiggestComponent(features, pairs, self.conf_thresh)
        if self.logger is not None:
            self.logger.info(
                f'Subset: kept {len(indices)} of {len(features)} images '
                f'(conf_thresh={self.conf_thresh}): {indices}')
        if len(indices) < 2:
            raise ValueError('Could not find at least two overlapping images')

        num_features = len(features)
        selected_features = [features[i] for i in indices]
        selected_pairs = []
        for i, original_i in enumerate(indices):
            selected_features[i].img_idx = i
            for j, original_j in enumerate(indices):
                pair = pairs[original_i * num_features + original_j]
                if pair.src_img_idx >= 0 and pair.dst_img_idx >= 0:
                    pair.src_img_idx = i
                    pair.dst_img_idx = j
                selected_pairs.append(pair)

        img_subset = [scaled_imgs[i] for i in indices]
        full_img_sizes_subset = [full_img_sizes[i] for i in indices]
        full_images_subset = [full_images[i] for i in indices]
        return selected_features, selected_pairs, img_subset, full_images_subset, full_img_sizes_subset
