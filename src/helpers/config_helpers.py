import json
import yaml
from io_helpers import search_files_by_extension


def parse_config(config_path):
    """
    @type config_path string
    """
    config_f = open(config_path)
    config = yaml.safe_load(config_f)
    config_f.close()
    if config is None:
        raise Exception('Empty config file')
    return config


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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
