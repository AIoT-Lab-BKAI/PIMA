from torch import nn
from models import ImageEncoder, ProjectionHead, ImageEncoderTimm, sentencesTransformer


class ImageTextMatching(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.image_encoder = ImageEncoderTimm(image_model_name=args.image_model_name,
                                              image_pretrained=args.image_pretrained, image_trainable=args.image_trainable)
        self.sentences_encoder = sentencesTransformer(
            model_name=args.text_model_name, trainable=args.text_trainable)
        self.image_projection = ProjectionHead(
            embedding_dim=args.image_embedding, projection_dim=args.projection_dim, dropout=args.dropout)
        self.text_projection = ProjectionHead(
            embedding_dim=args.text_embedding, projection_dim=args.projection_dim, dropout=args.dropout)

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
