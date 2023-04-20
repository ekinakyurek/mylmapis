from collections.abc import Iterable, Iterator
from typing import List, Union
import itertools


def batch(inputs: Union[List, Iterable], n: int):
    "Batch data into iterators of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")

    if not isinstance(inputs, Iterator):
        inputs = iter(inputs)

    while True:
        chunk_it = itertools.islice(inputs, n)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield itertools.chain((first_el,), chunk_it)

class SafeFormat(dict):
    def __missing__(self, key):
        print(key, " was missing in the formatter")
        return ""