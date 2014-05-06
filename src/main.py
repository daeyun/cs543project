import argparse
from helpers.config_helpers import *
from helpers.io_helpers import *
from prepare_data.crop_annotations import crop_annotations
from prepare_data.save_image_patch import save_image_patch


def main():
    parser = argparse.ArgumentParser(description='CS 543 Project')
    parser.add_argument('--config-file', nargs='?', default='../config.yaml',
                        help='path to config.yaml (default: ../config.yaml)')
    args = parser.parse_args()
    config_path = args.config_file

    try:
        config = parse_config(get_absolute_path(config_path))
    except Exception as e:
        pretty_print_exception("Could not load {}".format(config_path), e)
        return

    crop_annotations(config['paths']['input']['image'],
                     config['paths']['input']['annotation'],
                     config['paths']['output']['image patch'],
                     save_image_patch)

if __name__ == '__main__':
    main()