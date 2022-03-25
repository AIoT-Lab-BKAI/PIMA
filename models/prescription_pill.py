from torch import nn
import torch.nn.functional as F
from models import ImageEncoder, ProjectionHead, ImageEncoderTimm, sentencesTransformer, SBERTxSAGE
import torch


class PrescriptionPill(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.image_encoder = ImageEncoderTimm(image_model_name=args.image_model_name,
                                              image_pretrained=args.image_pretrained, image_trainable=args.image_trainable)

        # self.image_encoder = ImageEncoder()
        # if image_pretrained_link is not None:
        #     pre_train_state = torch.load(image_pretrained_link)
        #     image_model_state = self.image_encoder.state_dict()
        #     # 1. filter out unnecessary keys
        #     pretrained_dict = {k: v for k, v in pre_train_state.items() if k in image_model_state}
        #     # 2. overwrite entries in the existing state dict
        #     image_model_state.update(pretrained_dict)
        #     # 3. load the new state dict
        #     self.image_encoder.load_state_dict(image_model_state)
        #     print("Loaded pretrained image model successfully!")

        self.graph_encoder = SBERTxSAGE(
            input_dim=args.text_embedding, output_dim=args.graph_embedding, dropout_rate=args.dropout)

        self.sentences_encoder = sentencesTransformer(
            model_name=args.text_model_name, trainable=args.text_trainable)

        self.image_projection = ProjectionHead(
            embedding_dim=args.image_embedding, projection_dim=args.projection_dim, dropout=args.dropout)

        self.text_projection = ProjectionHead(
            embedding_dim=args.text_embedding, projection_dim=args.projection_dim, dropout=args.dropout)

        self.sentences_graph_projection = ProjectionHead(
            embedding_dim=args.text_embedding + args.graph_embedding, projection_dim=args.projection_dim, dropout=args.dropout)

        self.graph_post_process_layers = nn.Sequential(
            nn.BatchNorm1d(256, affine=False),
            nn.Dropout(p=args.dropout),
            nn.Linear(256, 2),  # 2 class: Drugname / Other
            nn.GELU()
        )

    def forward_graph(self, data, sentences_feature):
        # Getting graph embedding
        graph_features = self.graph_encoder(data, sentences_feature)
        # FOR KIE
        graph_extract = self.graph_post_process_layers(graph_features)
        graph_extract = F.log_softmax(graph_extract, dim=-1)
        return graph_extract, graph_features

    def get_image_features(self, image):
        x = self.image_encoder(image)
        x = self.image_projection(x)
        return x

    def forward(self, data):
        image_projection = self.get_image_features(data.pills_images)

        sentences_feature = self.sentences_encoder(
            data.text_sentences_ids, data.text_sentences_mask)
        # sentences_projection = self.text_projection(sentences_feature)
        graph_extract, graph_features = self.forward_graph(
            data, sentences_feature)

        sentences_graph_features = torch.cat(
            (sentences_feature, graph_features), dim=1)
        sentences_graph_features = self.sentences_graph_projection(
            sentences_graph_features)

        return image_projection, sentences_graph_features, graph_extract
