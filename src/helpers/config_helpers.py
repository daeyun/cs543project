import json
import os
import cv2
import yaml
from helpers.image_operation_helpers import rotate_image
from io_helpers import search_files_by_extension, path_to_filename
import re

__author__ = 'Daeyun Shin'


def parse_config(config_path):
    """
    @type config_path string
    """
    config_f = open(config_path)
    config = yaml.safe_load(config_f)
    config_f.close()
    if config is None:
        raise Exception('Empty config file')

    base_dir = config['paths']['base dir']

    def concat_base_dir(v):
        return os.path.join(base_dir, v)

    # modify paths in config
    operate_on_all_values(config['paths'], concat_base_dir)

    return config


def operate_on_all_values(dictionary, func):
    for k, v in dictionary.items():
        if isinstance(v, dict):
            operate_on_all_values(v, func)
        else:
            dictionary[k] = func(v)


def parse_annotations(json_path):
    """
    @type json_path: string
    @rtype: dict
    """
    annotations = {}
    annotation_paths = search_files_by_extension(json_path, ["json"])
    for annotation_path in annotation_paths:
        json_data = open(annotation_path)
        dict_data = json.load(json_data)

        rects = {}
        image_info = {}
        for c in dict_data['components']:
            if c['type'] == 'rectangle':
                label = c['label']
                if label not in rects:
                    rects[label] = []
                rects[label].append([c['x'], c['y'], c['w'], c['h']])
            elif c['type'] == 'image':
                image_info['orientation'] = c['orientation']
                image_info['w'] = c['w']
                image_info['h'] = c['h']

        annotations[dict_data['filename']] = {'rects': rects, 'image info': image_info}

    return annotations


def unpack_filename(filename):
    m = re.search('([^_/]+?)__([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([-0-9]+)_([-0-9]+).[^/_\.]+?$', filename)
    groups = m.groups()
    return tuple([groups[0]] + [int(g) for g in groups[1:]])


if __name__ == "__main__":
    import doctest

    doctest.testmod()