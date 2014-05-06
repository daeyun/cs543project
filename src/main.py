import argparse
import cv2


def main():
    parser = argparse.ArgumentParser(description='CS 543 Project')
    parser.add_argument('--config-file', nargs='?', default='../config.yaml',
                        help='path to config.yaml (default: ../config.yaml)')
    args = parser.parse_args()
    config_path = args.config_file


if __name__ == '__main__':
    main()