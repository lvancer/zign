from .dataset import zDataset


# from typing import TypeVar, Generic

# T = TypeVar('T')
# S = TypeVar('S')

# class Pair(Generic[T, S]):
#     def __init__(self, src: T, tgt: S):
#         self.src = src
#         self.tgt = tgt

#     def get_src(self) -> T:
#         return self.src

#     def get_tgt(self) -> S:
#         return self.tgt

#     def set_src(self, src: T) -> None:
#         self.src = src

#     def set_tgt(self, tgt: S) -> None:
#         self.tgt = tgt

#     def __repr__(self) -> str:
#         return f"Pair(src={self.src}, tgt={self.tgt})"