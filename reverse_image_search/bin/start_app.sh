image=reverse_image_search
# docker build -t $image -f reverse_image_search/Dockerfile reverse_image_search/
docker run -it --rm --name app -p 8501:8501 --network tritonserver -v $(pwd):/workspace $image streamlit run reverse_image_search/src/app.py
