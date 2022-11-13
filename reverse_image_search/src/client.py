import numpy as np
import tritonclient.http as httpclient


def create_triton_http_client():
    triton_client = httpclient.InferenceServerClient(
        url="triton_reverse_image_search:8000",
        verbose=True,
        # ssl=True,
        # ssl_options=ssl_options,
        # insecure=True,
        # ssl_context_factory=None,
    )
    return triton_client


def image_encoder(tritton_http_client, np_img):
    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput("INPUT__0", list(np_img.shape), "FP32"))
    inputs[0].set_data_from_numpy(np_img, binary_data=False)
    outputs.append(httpclient.InferRequestedOutput("OUTPUT__0", binary_data=True))
    results = tritton_http_client.infer(
        "image_encoder",
        inputs,
        outputs=outputs,
    )
    image_embedding = results.as_numpy("OUTPUT__0")
    return image_embedding


def vector_similarity(tritton_http_client, embedding_vec):
    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput("INPUT__0", list(embedding_vec.shape), "FP32"))
    inputs[0].set_data_from_numpy(embedding_vec, binary_data=False)
    outputs.append(httpclient.InferRequestedOutput("OUTPUT__0", binary_data=True))
    outputs.append(httpclient.InferRequestedOutput("OUTPUT__1", binary_data=True))
    results = tritton_http_client.infer(
        "vector_similarity",
        inputs,
        outputs=outputs,
    )
    nearest_neighbors = results.as_numpy("OUTPUT__0")
    similarities = results.as_numpy("OUTPUT__1")
    return nearest_neighbors, similarities


if __name__ == "__main__":
    client = create_triton_http_client()
    np_img = np.random.rand(1, 3, 224, 224).astype(np.float32)
    image_embedding = image_encoder(client, np_img)
    print(image_embedding.shape)
    nearest_neighbors, similarities = vector_similarity(client, image_embedding)
    print(nearest_neighbors.shape)
    print(similarities.shape)
    client.close()
