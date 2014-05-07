def assertTupleAlmostEqual(t1, t2, places=7):
    """
    @type t1: tuple
    @type t2: tuple
    @type places: int

    Doctest:
    >>> assertTupleAlmostEqual((0.1, 0.1), (0.10001, 0.10001), 4)
    >>> assertTupleAlmostEqual((0.1, 0.1), (0.10001, 0.10001), 5)
    Traceback (most recent call last):
        ...
    AssertionError: (0.1, 0.1) != (0.10001, 0.10001) within 5 places
    >>> assertTupleAlmostEqual((0.1, 0.1), (0.10001, 0), 2)
    Traceback (most recent call last):
        ...
    AssertionError: (0.1, 0.1) != (0.10001, 0) within 2 places
    """

    if t1 is None and t2 is None:
        return
    if t1 is None and t2 is not None:
        raise AssertionError("None != {}".format(t2))
    if t1 is not None and t2 is None:
        raise AssertionError("{} != None".format(t2))
    if round(abs(t1[0]-t2[0]), places) == 0 and \
                    round(abs(t1[1]-t2[1]), places) == 0:
        return
    raise AssertionError("{} != {} within {} places".format(t1, t2, places))

if __name__ == "__main__":
    import doctest

    doctest.testmod()