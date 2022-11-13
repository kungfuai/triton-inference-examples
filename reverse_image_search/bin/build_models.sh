# Build the model repository for Triton using a pre-trained model.
image=reverse_image_search
docker build -t $image -f reverse_image_search/Dockerfile reverse_image_search/
docker run -it --rm -v $(pwd):/workspace $image python -m reverse_image_search.src.build_models
