docker run -it --runtime runc --rm -v $(pwd):/opt/tritonserver/code nvcr.io/nvidia/tritonserver:22.09-py3 bash
# docker run -it --gpus all -v ${PWD}:/workspace/code nvcr.io/nvidia/pytorch:22.09-py3
