import numpy as np
import torch


class TorchVectorSimilarity(torch.nn.Module):

    """A class that computes vector similarity using PyTorch."""

    def __init__(self, db_vectors: np.ndarray, device: str = "cpu"):
        """Initializes the class.

        Args:
            vector_size: The size of the vectors.
            device: The device to use for the computation.
        """
        super().__init__()
        self.db_vectors = torch.from_numpy(db_vectors)
        self.db_vectors = torch.nn.functional.normalize(self.db_vectors, dim=1, p=2)
        self.device = device

    @torch.no_grad()
    def forward(self, vectors, k: int = 1) -> torch.Tensor:
        """Computes the similarity between the vectors and the database vectors.

        Args:
            vectors: The vectors to compare.

        Returns:
            The similarity between the vectors and the database vectors.
        """
        return self.nearest_neighbors(vectors, k=k)

    def nearest_neighbors(self, vectors: torch.Tensor, k: int = 1) -> torch.Tensor:
        """Finds the nearest neighbors of the vectors.

        Args:
            vectors: The vectors to find the nearest neighbors of.
            k: The number of nearest neighbors to find.

        Returns:
            The indices of the nearest neighbors.
        """
        vectors = torch.nn.functional.normalize(vectors, dim=1, p=2)
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


class AnnoyVectorSimilarity:
    def __init__(self, db_vectors):
        self.db_vectors = db_vectors
        self.index = AnnoyIndex(len(db_vectors[0]), "angular")
        for i, v in enumerate(db_vectors):
            self.index.add_item(i, v)
        self.index.build(10)

    def get_nns_by_vector(self, vector, n, search_k=-1, include_distances=False):
        if search_k == -1:
            search_k = len(self.vectors)
        ids, distances = self.index.get_nns_by_vector(
            vector, n, search_k, include_distances
        )
        return ids, distances

    def get_nns_by_item(self, i, n, search_k=-1, include_distances=False):
        if search_k == -1:
            search_k = len(self.vectors)
        ids, distances = self.index.get_nns_by_item(i, n, search_k, include_distances)
        return ids, distances

    def get_item_vector(self, i):
        return self.vectors[i]

    def get_n_items(self):
        return len(self.vectors)
