from os import makedirs, path
from typing import List
import numpy as np
import torch

TRITON_CONFIG = """
name: "vector_similarity"
backend: "pytorch"
max_batch_size : 0

input [
  {
    name: "INPUT__0"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]
output [
  {
    name: "OUTPUT__0"
    data_type: TYPE_INT32
    dims: [ -1 ]
  },
  {
    name: "OUTPUT__0"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]
"""


class TorchVectorSimilarity(torch.nn.Module):

    """A class that computes vector similarity using PyTorch."""

    def __init__(self, db_vectors: np.ndarray):
        """Initializes the class.

        Args:
            vector_size: The size of the vectors.
            device: The device to use for the computation.
        """
        super().__init__()
        self.db_vectors = torch.from_numpy(db_vectors)
        self.db_vectors = torch.nn.functional.normalize(self.db_vectors, dim=1, p=2)

    @torch.no_grad()
    def forward(self, vectors, k: int = 1) -> List[torch.Tensor]:
        """Computes the similarity between the vectors and the database vectors.

        Args:
            vectors: The vectors to compare.

        Returns:
            The similarity between the vectors and the database vectors.
        """
        return self.nearest_neighbors(vectors, k=k)

    def nearest_neighbors(
        self, vectors: torch.Tensor, k: int = 1
    ) -> List[torch.Tensor]:
        """Finds the nearest neighbors of the vectors.

        Args:
            vectors: The vectors to find the nearest neighbors of.
            k: The number of nearest neighbors to find.

        Returns:
            The indices of the nearest neighbors.
        """
        vectors = torch.nn.functional.normalize(vectors, dim=1, p=2.0)
        similarities = self.compute_similarity(vectors, self.db_vectors)
        return torch.topk(similarities, k=k, dim=1).indices, similarities

    def compute_similarity(
        self, vectors1: torch.Tensor, vectors2: torch.Tensor
    ) -> torch.Tensor:
        """Computes the similarity between two sets of vectors.

        Args:
            vectors1: The first set of vectors.
            vectors2: The second set of vectors.

        Returns:
            The similarity between the two sets of vectors.
        """
        # print(vectors1.shape)
        # print(vectors2.shape)
        return torch.mm(vectors1, vectors2.T)

    def save(self, output_file: str, use_trace: bool = True, save_triton_config=True):

        makedirs(path.dirname(output_file), exist_ok=True)
        if use_trace:
            image = torch.randn(1, 512)
            m = torch.jit.trace(self, image)
            m.save(output_file)
        else:
            m = torch.jit.script(self)
            torch.jit.save(m, output_file)

        if save_triton_config:
            with open(
                path.join(path.dirname(path.dirname(output_file)), "config.pbtxt"), "w"
            ) as f:
                f.write(TRITON_CONFIG)
