from .table import merge_columns, row_to_column


def test_merge():
    a = [[1, 2], [3, 4], [5, 6]]
    b = [[7], [8], [9]]
    ab = list(merge_columns(a, b))
    print(ab)
    assert ab == [[1, 2, 7], [3, 4, 8], [5, 6, 9]]


def test_merge_generator():
    a = [[1, 2], [3, 4], [5, 6]]
    b = dict(a=7, b=8, c=9)
    ab = list(merge_columns(a, row_to_column(b.values())))
    print(ab)
    assert ab == [[1, 2, 7], [3, 4, 8], [5, 6, 9]]
