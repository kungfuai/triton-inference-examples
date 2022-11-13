from glob import glob
import numpy as np
import os
from PIL import Image
import streamlit as st
from streamlit_image_select import image_select
from torchvision.datasets import FashionMNIST
from client import (
    image_encoder,
    vector_similarity,
    create_triton_http_client,
)


def prepare_example_images():
    test_ds = FashionMNIST("./data", download=True, train=False)
    image_paths = []
    for i in range(30):
        image_path = f"data/reverse_image_search/query/image{i}.png"
        if True or (not os.path.exists(image_path)):
            image = test_ds[i][0]
            # image = image.resize((28, 28))
            image = image.convert("RGB")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            image.save(image_path)
        image_paths.append(image_path)
    return image_paths


def main():
    st.title("Reverse Image Search")

    st.subheader("Pick an image to find for similar ones in the database")
    query_image_paths = prepare_example_images()
    query_image_path = image_select("Query Image", query_image_paths[18:])
    db_image_paths = glob("data/reverse_image_search/db/*.png")
    img = np.array(Image.open(query_image_path)).transpose(2, 0, 1)
    img = img.astype(np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    st.header("Similar images")
    st.write(f"Top 20 similar images out of {len(db_image_paths)}.")
    client = create_triton_http_client()
    embedding_vector = image_encoder(client, img)
    nearest_neighbors, scores = vector_similarity(client, embedding_vector)
    row_width = 5
    image_row = []
    for nn_idx, score in sorted(
        zip(nearest_neighbors[0], scores[0]), key=lambda x: x[1], reverse=True
    ):
        nn_image_path = f"data/reverse_image_search/db/{nn_idx}.png"
        nn_image = Image.open(nn_image_path)
        image_row.append(nn_image)
        if len(image_row) == row_width:
            for j, col in enumerate(st.columns(row_width)):
                col.image(image_row[j], caption=f"Score: {score:.2f}", width=100)
            image_row = []

    for j, col in enumerate(st.columns(row_width)):
        if j < len(image_row):
            col.image(image_row[j], width=100, caption=f"Score: {score:.2f}")


if __name__ == "__main__":
    main()
