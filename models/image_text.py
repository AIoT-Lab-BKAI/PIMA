from torch import nn
import torch
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

        self.W1 = nn.Linear(args.image_embedding, args.projection_dim)
        self.W2 = nn.Linear(args.image_embedding, args.projection_dim)

    def get_image_features(self, image):
        x = self.image_encoder(image)
        x = self.image_projection(x)
        return x

    def get_image_aggregation(self, image, label):
        x = self.image_encoder(image)
        # x = Wx + W mean( \x )
        # with all
        # x = self.W1(x) + self.W2(x.mean(dim=0))

        calculate_mean = torch.zeros(x.shape[0], x.shape[1]).cuda()
        for idx, value in enumerate(label):
            if sum(label != value) == 0:
                continue
            other_image = x[label != value].detach()
            calculate_mean[idx, :] = other_image.mean(dim=0)

        x = self.W1(x) + self.W2(calculate_mean)

        return x

    def get_sentences_features(self, data):
        x = self.sentences_encoder(
            data.text_sentences_ids, data.text_sentences_mask)
        x = self.text_projection(x)
        return x

    def forward(self, data):
        # return self.get_image_features(data.pills_images), self.get_sentences_features(data)
        return self.get_image_aggregation(data.pills_images, data.pills_images_labels), self.get_sentences_features(data)
