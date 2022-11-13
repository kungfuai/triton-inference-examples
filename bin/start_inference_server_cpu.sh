
tritonversion=22.10 # 21.10, 22.09, 22.01
image=nvcr.io/nvidia/tritonserver:${tritonversion}-py3
# Some alternatives:
# image=007439368137.dkr.ecr.us-east-2.amazonaws.com/sagemaker-tritonserver:22.07-py3-cpu # needs AWS credentials
# image=mcr.microsoft.com/azureml/tritonserver-21.06-py38-inference:latest # requires AVX instruction set
model_repo=$(pwd)/model_repo/

docker run -it --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $model_repo:/models \
    $image tritonserver --model-repository=/models

# To start the inference server on only one model:
# model_name=vector_similarity
# docker run -it --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $model_repo/$model_name:/models/$model_name \
#     $image tritonserver --model-repository=/models
