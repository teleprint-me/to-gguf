"""
to_gguf/tensors/raw.py
"""
from dataclasses import dataclass
from typing import Any, Iterable

import gguf
import numpy as np


@dataclass(frozen=True)
class RawTensor:
    """
    Represents a raw tensor without quantization.
    """

    name: str
    dtype: np.dtype[Any]
    valid_conversions: list[str]

    def tensor_to_bytes(self, n_elements: int) -> int:
        """
        Calculate the number of bytes required to store the tensor's data.

        Args:
            n_elements (int): Number of elements in the tensor.

        Returns:
            int: Number of bytes needed.
        """
        return n_elements * self.dtype.itemsize


@dataclass(frozen=True)
class RawQuantizedTensor(RawTensor):
    """
    Represents a raw tensor with quantization.
    """

    block_size: int
    quantized_dtype: np.dtype[Any]
    ggml_type: gguf.GGMLQuantizationType

    def _assert_block_size(self, n_elements: int) -> None:
        assert (
            n_elements % self.block_size == 0
        ), f"Invalid number of elements {n_elements} for {self.name} with block size {self.block_size}"

    def quantize(self, array: np.ndarray) -> np.ndarray:
        """
        Quantize the input array according to this tensor's quantization rules.

        Args:
            array (np.ndarray): Input array to be quantized.

        Returns:
            np.ndarray: Quantized array.
        """
        raise NotImplementedError(f"Quantization for {self.name} not implemented")

    def tensor_to_bytes(self, n_elements: int) -> int:
        """
        Calculate the number of bytes required to store the quantized tensor's data.

        Args:
            n_elements (int): Number of elements in the quantized tensor.

        Returns:
            int: Number of bytes needed.
        """
        self._assert_block_size(n_elements)
        return self.quantized_dtype.itemsize * (n_elements // self.block_size)


@dataclass(frozen=True)
class RawQ8_0Tensor(RawQuantizedTensor):
    """
    Represents a raw tensor with Q8_0 quantization.
    """

    def _assert_valid_array(self, array: np.ndarray) -> None:
        """
        Asserts that the input array is valid for Q8_0 quantization.

        Args:
            array (np.ndarray): Input array to be checked.

        Raises:
            AssertionError: If the array size or type is invalid.
        """
        assert (
            array.size % self.block_size == 0 and array.size != 0
        ), f"Invalid array size {array.size}"
        assert array.dtype == np.float32, f"Invalid array type {array.dtype}"

    def _quantize_blocks_q8_0(self, blocks: np.ndarray) -> Iterable[tuple[Any, Any]]:
        """
        Quantizes blocks of values using Q8_0 quantization.

        Args:
            blocks (np.ndarray): Blocks of input values to be quantized.

        Yields:
            Iterable[tuple[Any, Any]]: Yields tuples of scaling factors and quantized values.
        """
        scaling_factors = abs(blocks).max(axis=1) / np.float32(127)
        with np.errstate(divide="ignore"):
            quantized_blocks = (blocks / scaling_factors[:, None]).round()
        quantized_blocks[scaling_factors == 0] = 0
        yield from zip(scaling_factors, quantized_blocks)

    def quantize(self, array: np.ndarray) -> np.ndarray:
        """
        Quantize the input array using Q8_0 quantization.

        Args:
            array (np.ndarray): Input array to be Q8_0 quantized.

        Returns:
            np.ndarray: Q8_0 quantized array.
        """
        self._assert_valid_array(array)
        n_blocks = array.size // self.block_size
        blocks = array.reshape((n_blocks, self.block_size))
        return np.fromiter(
            self._quantize_blocks_q8_0(blocks),
            count=n_blocks,
            dtype=self.quantized_dtype,
        )
