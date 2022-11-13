import numpy as np
from os import makedirs, path
from tqdm import tqdm
from PIL import Image
import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from .image_encoder import ImageEncoder
from .vector_similarity import TorchVectorSimilarity


def train():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_ds = FashionMNIST("./data", download=True, train=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    db_vectors = []
    image_encoder = ImageEncoder(model_name="resnet18")
    model_path = "model_repo/image_encoder/1/model.pt"
    image_encoder.save(model_path)
    print("Saved image encoder model to", model_path)
    image_counter = 0
    for i, (images, _) in tqdm(enumerate(train_loader)):
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        for image in images:
            img_path = f"data/reverse_image_search/db/{image_counter}.png"
            makedirs(path.dirname(img_path), exist_ok=True)
            image = image.numpy().transpose(1, 2, 0)
            Image.fromarray((image / image.max() * 255).astype(np.uint8)).save(img_path)
            image_counter += 1
        vectors = image_encoder(images)
        if hasattr(vectors, "detach"):
            vectors = vectors.detach().numpy()
        else:
            vectors = vectors.numpy()
        db_vectors.append(vectors)
        if i >= 10:
            break
    db_vectors = np.concatenate(db_vectors, axis=0)
    print("DB vectors shape:", db_vectors.shape)
    vector_similarity = TorchVectorSimilarity(db_vectors)
    # save model
    model_path = "model_repo/vector_similarity/1/model.pt"
    vector_similarity.save(model_path)
    print("Vector similarity model saved to", model_path)


def load_predictor_from_numpy():
    db_vectors = np.load("db_vectors.npy")
    vector_similarity = TorchVectorSimilarity(db_vectors)
    return vector_similarity


def load_predictor():
    vector_similarity = torch.load("vector_similarity.pth")
    # vector_similarity = TorchVectorSimilarity(None)
    # vector_similarity.load_state_dict(torch.load("vector_similarity.pth"))
    return vector_similarity


def eval():
    vector_similarity = load_predictor()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    val_ds = FashionMNIST("./data", download=True, train=False, transform=transform)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # For visualization:
    train_ds = FashionMNIST("./data", download=True, train=True, transform=transform)

    for i, (images, _) in tqdm(enumerate(val_loader)):
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        vectors = ImageEncoder().encode(images)
        [ids, similarities] = vector_similarity.forward(vectors, k=3)
        print(ids.shape, similarities.shape)
        print(ids)
        # print(similarities)
        # visalize_reverse_image_search(images, ids, train_ds)
        if i >= 0:
            break


def visalize_reverse_image_search(images, ids, train_dataset):
    import matplotlib.pyplot as plt

    images = images.numpy()
    ids = ids.numpy()
    for i in range(min(4, images.shape[0])):
        fig, axs = plt.subplots(1, 4)
        axs[0].imshow(images[i].transpose(1, 2, 0))
        for j in range(3):
            axs[j + 1].imshow(train_dataset[ids[i, j]][0][0], cmap="gray")
        fig.savefig(f"reverse_image_search_{i}.png")


if __name__ == "__main__":
    train()
    # eval()
