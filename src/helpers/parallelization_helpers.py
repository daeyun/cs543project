__author__ = 'Daeyun Shin'


def chunks(l, n):
    """
    split list into n chunks
    :param l: List to divide into chunks
    :param n: Number of split lists
    http://stackoverflow.com/questions/2130016/splitting-a-list-of-arbitrary-size-into-only-roughly-n-equal-parts
    """
    avg = len(l) / float(n)
    out = []
    last = 0.0
    while last < len(l):
        out.append(l[int(last):int(last + avg)])
        last += avg
    return out
