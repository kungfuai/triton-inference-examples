This is a tutorial for using Triton to support a reverse image search application.

## Reverse image search

Reverse image search means finding similar images. An image is used as a query, rather than a text query like "cat photos".

Two major components are needed to support reverse image search:
- a image encoder module, which extract a feature vector from each image, and
- a vector similarity module, which returns the most similar images based on vector similarity.


To run this example (Docker needed), please checkout the repo:

```
git clone https://github.com/kungfuai/triton-inference-examples
cd triton-inference-examples
```

First, build the image search models:

```
bin/build_reverse_image_search_models.sh
```

Then, start the Triton inference server:

If you have GPU and CUDA available:

```
bash bin/start_inference_server_gpu.sh
```

If you want to use CPU only:

```
bash bin/start_inference_server_cpu.sh
```

The above steps will use docker images and pulling down the docker image can take some time.
