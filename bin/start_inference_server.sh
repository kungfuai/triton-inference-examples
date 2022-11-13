
tritonversion=21.10 # 22.09, 22.01
image=nvcr.io/nvidia/tritonserver:${tritonversion}-py3
model_repo=/home/zsi/projects/triton-inference-examples/model_repo

docker run -it --gpus '"device=0"' --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $model_repo:/models \
    $image tritonserver --model-repository=/models

# To start the inference server on only one model:
# model_name=vector_similarity
# docker run -it --gpus '"device=0"' --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $model_repo/$model_name:/models/$model_name \
#     $image tritonserver --model-repository=/models

# Set `--gpus all` to use all GPUs