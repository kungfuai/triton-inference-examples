import torchvision


class ImageEncoder:
    def __init__(self, model_name="resnet50"):
        self.model_name = model_name
        self.model = self._get_model()

    def _get_model(self):
        model = getattr(torchvision.models, self.model_name)(pretrained=True)
        model.eval()
        return model

    def encode(self, image):
        return self.model(image)
