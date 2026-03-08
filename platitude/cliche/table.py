from typing import Iterable


def merge_columns(*tables: list[Iterable]):
    columns = [iter(c) for c in tables]
    first = columns[0]
    for row in first:
        for col in columns[1:]:
            row += next(col)
        yield row


def row_to_column(row: Iterable):
    return ([r] for r in row)
