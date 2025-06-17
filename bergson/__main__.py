from simple_parsing import parse

from .build import build_index
from .data import IndexConfig

if __name__ == "__main__":
    build_index(parse(IndexConfig))
