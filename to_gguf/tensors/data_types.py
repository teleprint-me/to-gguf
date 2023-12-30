"""
to_gguf/tensors/data_types.py
"""
from dataclasses import dataclass
from typing import Any, Iterable

import gguf
import numpy as np


@dataclass(frozen=True)
class BasicTensorType:
    """
    Describes the fundamental data type of a tensor, focusing on its unquantized form.
    This class provides details about the data type's characteristics, including its name,
    numpy data type, and a list of valid types to which it can be converted.

    Attributes:
        name (str): The name of the tensor data type.
        dtype (np.dtype[Any]): The numpy data type of the tensor.
        valid_conversions (list[str]): A list of names of other data types to which
                                       this data type can be validly converted.

    Methods:
        tensor_to_bytes(n_elements: int) -> int: Calculates the number of bytes required
                                                 to store a specified number of elements
                                                 of this data type.
    """

    name: str
    dtype: np.dtype[Any]
    valid_conversions: list[str]

    def compute_byte_size(self, n_elements: int) -> int:
        """
        Calculate the number of bytes required to store the tensor's data.

        Args:
            n_elements (int): The number of elements in the tensor.

        Returns:
            int: The number of bytes needed to store the specified number of elements.
        """
        return n_elements * self.dtype.itemsize


@dataclass(frozen=True)
class QuantizedTensorType(BasicTensorType):
    """
    Describes the fundamental data type of a tensor, focusing on its quantized form.
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

    def compute_byte_size(self, n_elements: int) -> int:
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
class Q8_0TensorType(QuantizedTensorType):
    """
    Describes the fundamental data type of a tensor, focusing on its 8-bit quantized form.
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
