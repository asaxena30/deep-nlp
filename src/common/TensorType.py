from enum import Enum, unique


@unique
class TensorType(Enum):
    TORCH = "torch"
    NUMPY = "numpy"

