import numpy as np
from tqdm import tqdm
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from .image_encoder import ImageEncoder
from .vector_similarity import AnnoyVectorSimilarity, TorchVectorSimilarity


def train():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_ds = FashionMNIST("./data", download=True, train=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    db_vectors = []
    for i, (images, _) in tqdm(enumerate(train_loader)):
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        vectors = ImageEncoder().encode(images)
        if hasattr(vectors, "detach"):
            vectors = vectors.detach().numpy()
        else:
            vectors = vectors.numpy()
        db_vectors.append(vectors)
        if i >= 10:
            break
    db_vectors = np.concatenate(db_vectors, axis=0)
    print(db_vectors.shape)
    # save db_vectors to disk
    np.save("db_vectors.npy", db_vectors)


def load_predictor():
    db_vectors = np.load("db_vectors.npy")
    vector_similarity = TorchVectorSimilarity(db_vectors)
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
        ids, similarities = vector_similarity.forward(vectors, k=3)
        print(ids.shape, similarities.shape)
        print(ids)
        # print(similarities)
        visalize_reverse_image_search(images, ids, train_ds)
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
    # train()
    eval()
