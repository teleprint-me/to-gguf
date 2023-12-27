"""
to_gguf/tensors/abstract.py
"""

from abc import ABCMeta, abstractmethod

from to_gguf.tensors.data_types import BasicTensorType


class AbstractTensor(metaclass=ABCMeta):
    """
    Abstract base class representing a general tensor structure.

    This class defines the essential operations and transformations that can be
    performed on tensors, providing a template for concrete tensor implementations.
    Each tensor is associated with a specific data type descriptor, defining its
    fundamental characteristics such as data format and quantization.

    Attributes:
        tensor_type (BasicTensorType): Descriptor defining the tensor's data type.

    Methods:
        astype(tensor_type: BasicTensorType) -> Tensor: Converts the tensor to the specified data type.
        permute(n_head: int, n_head_kv: int) -> Tensor: Rearranges the tensor's dimensions.
        permute_part(n_part: int, n_head: int, n_head_kv: int) -> Tensor: Permutes a part of the tensor.
        part(n_part: int) -> Tensor: Extracts a part of the tensor.
        to_ggml() -> Tensor: Converts the tensor to a format compatible with GGML.
    """

    tensor_type: BasicTensorType

    @abstractmethod
    def astype(self, tensor_type: BasicTensorType) -> "AbstractTensor":
        """
        Converts the tensor to the specified data type.

        Args:
            tensor_type (BasicTensorType): The data type to convert the tensor to.

        Returns:
            Tensor: A new tensor of the specified data type.
        """
        ...

    @abstractmethod
    def permute(self, n_head: int, n_head_kv: int) -> "AbstractTensor":
        """
        Rearranges the tensor's dimensions based on the provided head and key/value head counts.

        Args:
            n_head (int): Number of attention heads.
            n_head_kv (int): Number of key/value pairs per head.

        Returns:
            Tensor: A new tensor with permuted dimensions.
        """
        ...

    @abstractmethod
    def permute_part(
        self, n_part: int, n_head: int, n_head_kv: int
    ) -> "AbstractTensor":
        """
        Permutes a part of the tensor based on the provided part index and head counts.

        Args:
            n_part (int): Part index to permute.
            n_head (int): Number of attention heads.
            n_head_kv (int): Number of key/value pairs per head.

        Returns:
            Tensor: A part of the tensor with permuted dimensions.
        """
        ...

    @abstractmethod
    def extract_part(self, n_part: int) -> "AbstractTensor":
        """
        Extracts a part of the tensor based on the provided part index.

        Args:
            n_part (int): Part index to extract.

        Returns:
            Tensor: The extracted part of the tensor.
        """
        ...

    @abstractmethod
    def to_ggml(self) -> "AbstractTensor":
        """
        Converts the tensor to a format compatible with GGML (Georgi Gerganov Machine Learning).

        Returns:
            Tensor: The tensor in a GGML-compatible format.
        """
        ...
