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

    return parser.parse_args()


def main():
    args = cli_argument_parser()
    try:
        config = YamlReader(logger, args.config).get_config()[0]
        images = read_images(logger, config['images_dir'])
        stitcher = PanoramaStitcher.create(
            logger, config.get('stitcher_type', 'DetailedStitcher'), config)
        panorama = stitcher.stitch(images)
        save_image(logger, panorama, config.get('out_file_name', 'panorama.jpg'))
        if getattr(stitcher, 'perf_report', None) is not None:
            report_file_name = config.get('report_file_name', 'stitching_report.yaml')
            YamlReader(logger).save(stitcher.perf_report, report_file_name)
    except Exception as ex:
        logger.error((f'{str(ex)}'))


if __name__ == '__main__':
    sys.exit(main() or 0)
