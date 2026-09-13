import sys
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from stitching.read_write_data import YamlReader, read_images, save_image  # noqa: E402
from stitching.panorama_stitcher import PanoramaStitcher  # noqa: E402

logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger()


def cli_argument_parser():
    parser = argparse.ArgumentParser('Panorama stitching sample')

    parser.add_argument('-c', '--config',
                        type=str,
                        dest='config',
                        help='Path for yaml config for detailed stitching')
    parser.add_argument('-t', '--stitcher_type',
                        type=str,
                        choices=['OpenCVStitcher',
                                 'SuperPointLightGlueStitcher',
                                 'DetailedStitcher'],
                        default='OpenCVStitcher',
                        dest='stitcher_type',
                        help='Stitcher type')
    parser.add_argument('-d', '--images_dir',
                        type=str,
                        default='../images/kitchen',
                        dest='images_dir',
                        help='Directory that contains sequence of indoor images')
    parser.add_argument('-kp', '--keypoints_num',
                        help='Number of keypoints (for SuperPoint)',
                        type=int,
                        dest='keypoints_num',
                        default=2048)
    parser.add_argument('-o', '--out_file_name',
                        type=str,
                        default='panorama.jpg',
                        dest='out_file_name',
                        help='File name to save stitched panorama')

    return parser.parse_args()


def get_parameters_dict(args):
    yaml_reader = YamlReader(logger, args.config)
    return yaml_reader.get_config()[0]


def main():
    args = cli_argument_parser()
    try:
        images = read_images(logger, args.images_dir)
        stitcher = PanoramaStitcher.create(logger, args.stitcher_type, get_parameters_dict(args))
        panorama = stitcher.stitch(images)
        save_image(logger, panorama, args.out_file_name)
    except Exception as ex:
        logger.error((f'{str(ex)}'))


if __name__ == '__main__':
    sys.exit(main() or 0)
