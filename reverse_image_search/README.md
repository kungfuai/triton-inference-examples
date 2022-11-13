This is a tutorial for using Triton to support a reverse image search application.

## Reverse image search

Reverse image search means finding similar images. An image is used as a query, rather than a text query like "cat photos".

Two major components are needed to support reverse image search:
- a image encoder module, which extract a feature vector from each image, and
- a vector similarity module, which returns the most similar images based on vector similarity.


To run this example, please checkout the repo:

```
git clone https://github.com/kungfuai/triton-inference-examples
cd triton-inference-examples
```

First, build the image search models:

```
bin/build_reverse_image_search_models.sh
```

Then, start the Triton inference server by:

```
bash bin/start_inference_server.sh
```