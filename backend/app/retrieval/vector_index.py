"""Exact cosine-similarity search over an in-memory NumPy matrix.

At this scale, exact search is fast and simple, and this class is the place
that would change if a different backend were introduced later.
"""

import numpy as np


class VectorIndex:
    """An in-memory, exact cosine-similarity index over chunk embeddings."""

    def __init__(self, dim: int):
        """Creates an empty index for embeddings of a given dimension.

        Args:
            dim: The dimension of the embeddings this index will hold.
        """
        self._dim = dim
        self._ids: list[int] = []
        self._matrix = np.zeros((0, dim), dtype=np.float32)

    @property
    def size(self) -> int:
        """Returns the number of embeddings currently in the index.

        Returns:
            The number of embeddings currently in the index.
        """
        return len(self._ids)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """L2-normalises a vector while guarding against division by zero.

        Args:
            vector: A vector to normalise.

        Returns:
            `vector` scaled to unit length, or the original vector if it is
            the zero vector.
        """
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def add(self, chunk_id: int, embedding: list[float]) -> None:
        """Adds one chunk's embedding to the index.

        For adding several embeddings at once, use `add_many` instead:
        each call here reallocates and copies the whole matrix, so adding
        N embeddings one at a time here costs O(N^2), not O(N).

        Args:
            chunk_id: The chunk's id.
            embedding: The embedding vector to store.
        """
        self.add_many([chunk_id], [embedding])

    def add_many(self, chunk_ids: list[int], embeddings: list[list[float]]) -> None:
        """Adds several chunk embeddings to the index in one batch.

        Appends every embedding with a single matrix reallocation instead of
        one per embedding (see `add`). Use this for anything but a single
        chunk: adding thousands one at a time is O(N^2) and made startup slow.

        Args:
            chunk_ids: The chunk ids.
            embeddings: Their embedding vectors, in the same order as
                `chunk_ids`.
        """
        if not chunk_ids:
            return
        matrix = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized = np.divide(matrix, norms, out=matrix.copy(), where=norms > 0)
        self._ids.extend(chunk_ids)
        self._matrix = np.vstack([self._matrix, normalized])

    def remove(self, chunk_ids: set[int]) -> None:
        """Removes embeddings for the given chunk ids, if present.

        This keeps the in-memory index aligned with the database when a report
        is deleted or reset.

        Args:
            chunk_ids: The chunk ids to remove. Ids not currently in the index
                are ignored.
        """
        if not chunk_ids:
            return
        keep_mask = [chunk_id not in chunk_ids for chunk_id in self._ids]
        self._ids = [chunk_id for chunk_id, keep in zip(self._ids, keep_mask, strict=True) if keep]
        self._matrix = self._matrix[keep_mask]

    def search(
        self, query_embedding: list[float], candidate_ids: list[int] | None, top_k: int
    ) -> list[tuple[int, float]]:
        """Finds the chunks most similar to a query embedding.

        Args:
            query_embedding: The query's embedding vector.
            candidate_ids: Restrict the search to these chunk ids (e.g.
                chunks scoped to certain reports), or None to search every
                indexed chunk.
            top_k: The maximum number of results to return.

        Returns:
            `(chunk_id, cosine_score)` pairs, best match first. Empty if the
            index is empty or no indexed chunk is in `candidate_ids`.
        """
        if self.size == 0:
            return []
        query = self._normalize(np.asarray(query_embedding, dtype=np.float32))

        if candidate_ids is not None:
            candidate_set = set(candidate_ids)
            mask = [chunk_id in candidate_set for chunk_id in self._ids]
            if not any(mask):
                return []
            indices = np.nonzero(mask)[0]
            matrix = self._matrix[indices]
            ids = [self._ids[i] for i in indices]
        else:
            matrix = self._matrix
            ids = self._ids

        scores = matrix @ query
        order = np.argsort(-scores)[:top_k]
        return [(ids[i], float(scores[i])) for i in order]
