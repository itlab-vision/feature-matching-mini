import os
import cv2
import importlib
from pathlib import Path


def read_images(logger, images_dir):
    logger.info(f'START: Read images from {images_dir}')
    imgs = []
    files = [os.path.join(images_dir, f)
             for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    logger.info(f'Available images: {files}')
    for file in files:
        logger.info(f'Read image {file}')
        img = cv2.imread(file)
        if img is None:
            raise Exception(f'Error when reading image {file}')
        imgs.append(img)
    logger.info('FINISH: Read images')
    return imgs


def save_image(logger, image, filename):
    logger.info(f'START: Save image to {filename}')
    if filename is not None:
        cv2.imwrite(filename, image)
    else:
        raise ValueError('File name to save image is None')
    logger.info('FINISH: Save image')


class YamlReader:
    def __init__(self, logger, config_path=None):
        self.yaml = importlib.import_module('yaml')
        self.logger = logger
        if config_path is not None:
            self.logger.info(f'Getting config from {config_path}')
            if (Path(config_path).exists()):
                with open(config_path, 'r') as file:
                    self.config = self.yaml.safe_load(file)
            else:
                raise ValueError(f'File {config_path} does not exists.')

    def get_config(self):
        return self.config

    def save(self, dictionary, path):
        self.logger.info(f'Dumping dictionary to {path}')
        with open(path, 'w') as file:
            self.yaml.dump(dictionary, file)

    def read(self, path):
        with open(path, 'r') as file:
            self.config = self.yaml.safe_load(file)


def read_image(logger, image_path):
    logger.info(f'START: Read image from {image_path}')
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f'Error when reading image {image_path}')
    logger.info('FINISH: Read image')
    return img


def save_model(logger, model, filename):
    import open3d
    logger.info(f'START: Save model to {filename}')
    if filename is not None:
        open3d.io.write_triangle_mesh(filename, model, write_triangle_uvs=True)
        logger.info('FINISH: Save model')
    else:
        raise ValueError('File name to save model is None')
