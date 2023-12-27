"""
to_gguf/tensors/concrete.py
"""

import numpy as np

from to_gguf.tensors.abstract import AbstractTensor
from to_gguf.tensors.constants import NUMPY_TYPE_TO_TENSOR_DATA_TYPE, TensorDataType
from to_gguf.tensors.data_types import BasicTensorType


class Tensor(AbstractTensor):
    def __init__(self, ndarray: np.ndarray) -> None:
        assert isinstance(ndarray, np.ndarray)
        self.ndarray = ndarray
        # Determine the current tensor type based on its numpy dtype
        self.current_tensor_type = NUMPY_TYPE_TO_TENSOR_DATA_TYPE[ndarray.dtype]

    def astype(self, target_tensor_type: BasicTensorType) -> "Tensor":
        """
        Converts the tensor to the specified target tensor type.

        Args:
            target_tensor_type (BasicTensorType): The target tensor type to convert to.

        Returns:
            Tensor: A new tensor of the target tensor type.
        """
        # If the current tensor type is BF16, first convert it to FP32
        if self.current_tensor_type == TensorDataType.BF16:
            self.ndarray = self._bf16_to_fp32(self.ndarray)

        # Convert the ndarray to the dtype of the target tensor type
        return Tensor(self.ndarray.astype(target_tensor_type.dtype))

    def permute(self, n_head: int, n_head_kv: int) -> "Tensor":
        return Tensor(self._permute(self.ndarray, n_head, n_head_kv))

    def to_ggml(self) -> "Tensor":
        return self

    def permute_part(self, n_part: int, n_head: int, n_head_kv: int) -> "Tensor":
        r = self.ndarray.shape[0] // 3
        return Tensor(
            self._permute(
                self.ndarray[r * n_part : r * n_part + r, ...], n_head, n_head_kv
            )
        )

    def extract_part(self, n_part: int) -> "Tensor":
        r = self.ndarray.shape[0] // 3
        return Tensor(self.ndarray[r * n_part : r * n_part + r, ...])

    def _bf16_to_fp32(self, bf16_arr: np.ndarray) -> np.ndarray:
        """
        Convert BF16 format to FP32 format.

        Args:
            bf16_arr (np.ndarray): Input array in BF16 format.

        Returns:
            np.ndarray: Array converted to FP32 format.
        """
        assert (
            bf16_arr.dtype == np.uint16
        ), f"Expected uint16 dtype, got {bf16_arr.dtype}"
        fp32_arr = bf16_arr.astype(np.uint32) << 16
        return fp32_arr.view(np.float32)

    def _permute(self, weights: np.ndarray, n_head: int, n_head_kv: int) -> np.ndarray:
        """
        Rearrange the dimensions of a tensor for the transformer model.

        Args:
            weights (np.ndarray): Input tensor.
            n_head (int): Number of attention heads.
            n_head_kv (int): Number of key/value pairs in attention heads.

        Returns:
            np.ndarray: Tensor with rearranged dimensions.
        """
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (
            weights.reshape(
                n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:]
            )
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )
