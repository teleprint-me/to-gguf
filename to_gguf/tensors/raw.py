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

    def quantize(self, arr: np.ndarray) -> np.ndarray:
        """
        Quantize the input array according to this tensor's quantization rules.

        Args:
            arr (np.ndarray): Input array to be quantized.

        Returns:
            np.ndarray: Quantized array.
        """
        raise NotImplementedError(f"Quantization for {self.name} not implemented")

    def elements_to_bytes(self, n_elements: int) -> int:
        """
        Calculate the number of bytes required to store the quantized tensor's data.

        Args:
            n_elements (int): Number of elements in the quantized tensor.

        Returns:
            int: Number of bytes needed.
        """
        assert (
            n_elements % self.block_size == 0
        ), f"Invalid number of elements {n_elements} for {self.name} with block size {self.block_size}"
        return self.quantized_dtype.itemsize * (n_elements // self.block_size)


@dataclass(frozen=True)
class RawQ8_0Tensor(RawQuantizedTensor):
    """
    Represents a raw tensor with Q8_0 quantization.
    """

    def quantize(self, arr: np.ndarray) -> np.ndarray:
        """
        Quantize the input array using Q8_0 quantization.

        Args:
            arr (np.ndarray): Input array to be Q8_0 quantized.

        Returns:
            np.ndarray: Q8_0 quantized array.
        """
        assert (
            arr.size % self.block_size == 0 and arr.size != 0
        ), f"Bad array size {arr.size}"
        assert arr.dtype == np.float32, f"Bad array type {arr.dtype}"
        n_blocks = arr.size // self.block_size
        blocks = arr.reshape((n_blocks, self.block_size))

        def quantize_blocks_q8_0(blocks: np.ndarray) -> Iterable[tuple[Any, Any]]:
            d = abs(blocks).max(axis=1) / np.float32(127)
            with np.errstate(divide="ignore"):
                qs = (blocks / d[:, None]).round()
            qs[d == 0] = 0
            yield from zip(d, qs)

        return np.fromiter(
            quantize_blocks_q8_0(blocks), count=n_blocks, dtype=self.quantized_dtype
        )
