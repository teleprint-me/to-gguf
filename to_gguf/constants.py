"""
to_gguf/constants.py

Centralized constants and mappings for model conversion processes.
"""
from enum import Enum
from typing import Dict, NamedTuple, Tuple

import numpy as np
from gguf.constants import MODEL_ARCH


class DataTypeTuple(NamedTuple):
    dtype: np.dtype
    valid_conversions: list[str]


class DataType(Enum):
    F16 = DataTypeTuple(np.float16, ["F32", "Q8_0"])
    F32 = DataTypeTuple(np.float32, ["F16", "Q8_0"])
    I32 = DataTypeTuple(np.int16, [])
    BF16 = DataTypeTuple(np.uint16, ["F32", "F16", "Q8_0"])
    I8 = DataTypeTuple(np.dtype([("d", "<f2"), ("qs", "i1", (32,))]), [])  # Q8_0


# GGML File Type to Data Type Mapping
class GGMLFileType(Enum):
    # NOTE: GGMLFileType's are referenced by index value.
    # e.g., AllF32 is 0, MostlyF16 is 1, and MostlyQ8_0 is 7
    # This mapping affects processing and execution.
    # Need to solve this problem sooner than later.
    AllF32 = DataType.F32
    MostlyF16 = DataType.F16  # Except 1D tensors
    MostlyQ8_0 = DataType.I8  # Except 1D tensors

    def type_for_tensor(self, name: str, tensor_shape: Tuple[int, ...]) -> DataType:
        """Determine the appropriate DataType for a given tensor in the file."""
        # 1D tensors are always F32.
        return self.value if len(tensor_shape) > 1 else DataType.F32.value


# Set the LLaMa Architecture
LLAMA_ARCH = MODEL_ARCH.LLAMA

# Default number of threads for concurrent operations
DEFAULT_CONCURRENCY = 8

# Mapping of NumPy Types to Data Types
NUMPY_TYPE_TO_DATA_TYPE: Dict[str, DataType] = {dt.name: dt for dt in DataType}

# Mapping of GGML Types to Data Types
GGML_FILE_TYPE_TO_DATA_TYPE: Dict[str, DataType] = {
    ft.name: ft.value for ft in GGMLFileType
}

# Mapping of Safetensors to Data Types
SAFETENSORS_DATA_TYPES: Dict[str, DataType] = {
    "BF16": DataType.BF16,
    "F16": DataType.F16,
    "F32": DataType.F32,
    "I32": DataType.I32,
    "I8": DataType.I8,  # For Q8_0
}
