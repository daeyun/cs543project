import inspect
import os
import errno
import re
import sys
import cv2


def make_sure_dir_exists(path):
    """
    Try to create directories. If they already exist, ignore the error.
    @type path: string
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def get_absolute_path(path):
    """
    Concatenate path with the caller's absolute path.
    @type path: string
    """
    if not path.startswith((os.sep, '~')):
        path = os.path.join(os.path.dirname(os.path.realpath(inspect.getfile(sys._getframe(1)))), path)
    return path


def indented_print(msg, indentation='\t'):
    """
    @type msg: string
    """
    msg_string = str(msg)
    lines = msg_string.split('\n')
    for line in lines:
        print '\t' + line


def pretty_print_exception(msg, e):
    """
    @type msg: string
    @type e: Exception
    """
    print "{}\n[{}]".format(msg, type(e).__name__)
    indented_print(str(e))


def search_files_by_extension(path, extensions):
    """
    Find all files in a directory with given file extensions.

    @type path: string
    @type extensions: list
    @rtype: list
    """
    file_names = []
    path = os.path.normpath(path) + os.sep
    search_pattern = '.({})$'.format('|'.join(extensions))
    for file_name in os.listdir(path):
        if re.search(search_pattern, file_name, re.IGNORECASE):
            file_names.append(path + file_name)
    return file_names


def path_to_filename(path):
    """
    >>> path_to_filename('/home/user/hello.txt')
    'hello.txt'
    >>> path_to_filename('~/hello.txt')
    'hello.txt'
    >>> path_to_filename('hello.txt')
    'hello.txt'
    """
    return path.split('/')[-1]


def save_image(image, out_path):
    """
    @type image: ndarray
    @type out_path: string
    """
    cv2.imwrite(image, out_path)
