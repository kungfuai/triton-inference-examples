This is a tutorial for using Triton to support a reverse image search application.

## Reverse image search

Reverse image search means finding similar images. An image is used as a query, rather than a text query like "cat photos".

Two major components are needed to support reverse image search:
- a image encoder module, which extract a feature vector from each image, and
- a vector similarity module, which returns the most similar images based on vector similarity.

