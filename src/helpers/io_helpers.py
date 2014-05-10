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
        print indentation + line


def pretty_print_exception(msg, e):
    """
    @type msg: string
    @type e: Exception
    """
    print "{}\n[{}]".format(msg, type(e).__name__)
    indented_print(str(e))


def search_files_by_extension(path, extensions, is_recursive=False):
    """
    Find all files in a directory with given file extensions.

    @type path: string
    @type extensions: list
    @rtype: list
    """
    filenames = []
    path = os.path.normpath(path) + os.sep
    search_pattern = '.({})$'.format('|'.join(extensions))

    if is_recursive:
        # list of full paths
        paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(path) for f in fn]
    else:
        # list of the names of the files
        paths = os.listdir(path)

    for filename in paths:
        if re.search(search_pattern, filename, re.IGNORECASE):
            if not is_recursive:
                filenames.append(path + filename)
            else:
                filenames.append(filename)
    return filenames


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


def save_image(out_path, image):
    """
    @type image: ndarray
    @type out_path: string
    """
    cv2.imwrite(out_path, image)


def ensure_extension(filename, ext):
    """
    Make sure filename ends with .ext. Add .ext if not.
    """
    if ext.startswith('.'):
        ext = ext[1:]

    search_pattern = '\.{}$'.format(ext)
    if re.search(search_pattern, filename, re.IGNORECASE) is None:
        return "{}.{}".format(filename, ext)
    return filename
