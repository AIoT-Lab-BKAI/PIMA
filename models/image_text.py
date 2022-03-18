from torch import nn
import config as CFG
from models import ImageEncoder, ProjectionHead, ImageEncoderTimm, sentencesTransformer


class ImageTextMatching(nn.Module):
    def __init__(self, image_embedding=CFG.image_embedding, text_embedding=CFG.text_embedding):
        super().__init__()
        self.image_encoder = ImageEncoderTimm()
        self.sentences_encoder = sentencesTransformer()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)

    def get_image_features(self, image):
        x = self.image_encoder(image)
        x = self.image_projection(x)
        return x

    def get_sentences_features(self, data):
        x = self.sentences_encoder(
            data.text_sentences_ids, data.text_sentences_mask)
        x = self.text_projection(x)
        return x

    def forward(self, data):
        return self.get_image_features(data.pills_images), self.get_sentences_features(data)
