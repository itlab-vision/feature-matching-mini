import numpy as np
from pathlib import Path

from src.image_utils import read_image


class HPatchesDataManager:
    _ref_filename = "1.ppm"
    _img_indices = [2, 3, 4, 5, 6]

    def __init__(self, logger, config=None):
        if config is None:
            config = {}

        self._raw_data_path = Path(config.pop("raw_data_path", "hpatches-sequences-release"))
        self._num_scenes = config.pop("num_scenes", 116)
        self._scenes_batch_size = config.pop("scenes_batch_size", 4)
        self._logger = logger

        self._current_idx = 0
        self.all_scenes = self._get_all_scenes()

    def _get_all_scenes(self):
        scenes = [d for d in self._raw_data_path.iterdir() if d.is_dir()]
        if self._num_scenes:
            scenes = scenes[:self._num_scenes]
        return scenes

    def has_more_data(self):
        return self._current_idx < len(self.all_scenes)

    def _load_single_scene(self, scene_dir):
        ref_path = scene_dir / self._ref_filename
        img_ref = read_image(ref_path)

        if img_ref is None:
            self._logger.warning(f"Could not read reference image in {scene_dir}")
            return None

        scene_data = {
            'ref_img': img_ref,
            'ref_shape': img_ref.shape,
            'targets': {},
            'name': scene_dir.name
        }

        for i in self._img_indices:
            target_path = scene_dir / f"{i}.ppm"
            img_target = read_image(target_path)
            h_path = scene_dir / f"H_1_{i}"

            if img_target is not None and h_path.exists():
                H = np.loadtxt(str(h_path))
                scene_data['targets'][i] = {
                    'image': img_target,
                    'H': H,
                    'tgt_shape': img_target.shape
                }
        return scene_data

    def load_batch(self):
        if not self.has_more_data():
            self._logger.info("No more scenes to process.")
            return None

        if not self._current_idx:
            self._logger.info(f"Loading {self._num_scenes} scenes")

        start = self._current_idx
        end = min(start + self._scenes_batch_size, len(self.all_scenes))

        batch_paths = self.all_scenes[start: end]
        batch_data = {}

        for scene_dir in batch_paths:
            self._logger.info(f"Loading {scene_dir}")
            scene_content = self._load_single_scene(scene_dir)
            if scene_content:
                batch_data[scene_dir.name] = scene_content

        self._current_idx = end
        if self._current_idx == len(self.all_scenes):
            self._logger.info(f"Loaded {self._num_scenes} scenes")

        return batch_data

    def reset(self):
        self._current_idx = 0
