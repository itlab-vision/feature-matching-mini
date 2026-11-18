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
    parser = argparse.ArgumentParser(description='Stitch overlapping images into a panorama')

    parser.add_argument('-c', '--config', required=True, help='Path to the YAML configuration')
    parser.add_argument('--images-dir', help='Override the input directory from the configuration')
    parser.add_argument('--output', help='Override the output panorama path')
    parser.add_argument('--report', help='Override the YAML report path')

    return parser.parse_args()


def main():
    args = cli_argument_parser()
    try:
        config = YamlReader(logger, args.config).get_config()
        if isinstance(config, list):
            config = config[0]
        if args.images_dir is not None:
            config['images_dir'] = args.images_dir
        if args.output is not None:
            config['out_file_name'] = args.output
        if args.report is not None:
            config['report_file_name'] = args.report
        images = read_images(logger, config['images_dir'])
        stitcher = PanoramaStitcher.create(
            logger, config.get('stitcher_type', 'DetailedStitcher'), config)
        panorama = stitcher.stitch(images)
        save_image(logger, panorama, config.get('out_file_name', 'panorama.jpg'))
        if getattr(stitcher, 'perf_report', None) is not None:
            report_file_name = config.get('report_file_name', 'stitching_report.yaml')
            YamlReader(logger).save(stitcher.perf_report, report_file_name)
    except Exception as error:
        logger.error('%s', error)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
