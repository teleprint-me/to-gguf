"""
to_gguf/constants.py

Centralized constants and mappings for model conversion processes.
"""

from enum import Enum, IntEnum
from typing import Dict, Tuple

import gguf
import numpy as np

from to_gguf.tensors.data_types import BasicTensorType, Q8_0TensorType


class TensorDataType(Enum):
    F16 = BasicTensorType(
        name="F16",
        dtype=np.float16,
        valid_conversions=["F32", "Q8_0"],
    )
    F32 = BasicTensorType(
        name="F32",
        dtype=np.float32,
        valid_conversions=["F16", "Q8_0"],
    )
    I32 = BasicTensorType(
        name="I32",
        dtype=np.int16,
        valid_conversions=[],
    )
    BF16 = BasicTensorType(
        name="BF16",
        dtype=np.uint16,
        valid_conversions=["F32", "F16", "Q8_0"],
    )
    I8 = Q8_0TensorType(  # Inherits from BasicTensorType
        name="Q8_0",
        dtype=np.dtype(np.float32),
        valid_conversions=[],
        block_size=32,
        quantized_dtype=np.dtype([("d", "<f2"), ("qs", "i1", (32,))]),
        ggml_type=gguf.GGMLQuantizationType.Q8_0,
    )


# GGML File Type to Data Type Mapping
class GGMLFileType(IntEnum):
    AllF32 = 0
    MostlyF16 = 1  # Except 1D Tensors
    MostlyQ8_0 = 7  # Except 1D Tensors

    def type_for_tensor(self, tensor_shape: Tuple[int, ...]) -> BasicTensorType:
        """Determine the appropriate DataType for a given tensor in the file.

        Args:
            tensor_shape (Tuple[int, ...]): Shape of the tensor.

        Returns:
            TensorDataType: Appropriate data type for the tensor.

        Note:
            1D tensors are always treated as F32 regardless of the file type.
        """
        if len(tensor_shape) > 1:
            if self == GGMLFileType.AllF32:
                return TensorDataType.F32
            elif self == GGMLFileType.MostlyF16:
                return TensorDataType.F16
            elif self == GGMLFileType.MostlyQ8_0:
                return TensorDataType.I8
        return TensorDataType.F32


# Set the LLaMa Architecture
LLAMA_ARCH = gguf.constants.MODEL_ARCH.LLAMA

# Default number of threads for concurrent operations
DEFAULT_CONCURRENCY = 8

# Mapping of NumPy Types to Data Types
NUMPY_TYPE_TO_TENSOR_DATA_TYPE: Dict[np.dtype, TensorDataType] = {
    dt.dtype: dt for dt in TensorDataType
}

# Safetensors data types mapping
SAFETENSORS_DATA_TYPES: Dict[str, TensorDataType] = {
    "BF16": TensorDataType.BF16,
    "F16": TensorDataType.F16,
    "F32": TensorDataType.F32,
    "I32": TensorDataType.I32,
    "I8": TensorDataType.Q8_0,
}
