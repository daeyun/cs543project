import argparse

from helpers.config_helpers import parse_config
from helpers.io_helpers import pretty_print_exception, get_absolute_path
from prepare_data.extract_labeled_images import extract_labeled_images
from prepare_data.extract_noise import extract_noise
from prepare_data.extract_square_images import extract_square_images
from prepare_data.save_image_patch import save_image_patch

__author__ = 'Daeyun Shin'


def main():
    parser = argparse.ArgumentParser(description='CS 543 Project')
    parser.add_argument('task')
    parser.add_argument('--config-file', nargs='?', default='../config.yaml',
                        help='path to config.yaml (default: ../config.yaml)')
    args = parser.parse_args()

    config_path = args.config_file

    try:
        config = parse_config(get_absolute_path(config_path))
        # g_config is a global variable
        global g_config
        g_config = config
    except Exception as e:
        pretty_print_exception("Could not load {}".format(config_path), e)
        return

    if args.task == 'extract-labeled-images':
        input_image_dir = config['paths']['input']['initial']['image']
        input_annotation_dir = config['paths']['input']['initial']['annotation']
        out_dir = config['paths']['output']['data prep']['image']['positive']

        print """Extracting labeled image patches
        Image directory: {}
        Annotation directory: {}
        Output directory: {}""".format(input_image_dir, input_annotation_dir, out_dir)

        extract_labeled_images(input_image_dir, input_annotation_dir, out_dir, save_image_patch)

    elif args.task == 'extract-square-images':
        input_image_dir = config['paths']['input']['initial']['image']
        input_annotation_dir = config['paths']['input']['initial']['annotation']
        out_dir = config['paths']['output']['data prep']['square image']['positive']

        extract_square_images(input_image_dir, input_annotation_dir, out_dir, callback=save_image_patch, max_side=256)

    elif args.task == 'generate-negative-samples':
        input_image_dir = config['paths']['input']['initial']['image']
        input_annotation_dir = config['paths']['input']['initial']['annotation']
        out_dir = config['paths']['output']['data prep']['square image']['negative']

        extract_noise(input_image_dir, input_annotation_dir, out_dir, save_image_patch)

if __name__ == '__main__':
    main()