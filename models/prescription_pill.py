from torch import nn
import torch.nn.functional as F
from models import ImageEncoder, ProjectionHead, ImageEncoderTimm, sentencesTransformer, SBERTxSAGE
import torch


class PrescriptionPill(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.image_encoder = ImageEncoder(model_name=args.image_model_name,
                                              pretrained=args.image_pretrained, trainable=args.image_trainable)

        self.graph_encoder = SBERTxSAGE(
            input_dim=args.text_embedding, output_dim=args.graph_embedding, dropout_rate=args.dropout)

        self.sentences_encoder = sentencesTransformer(
            model_name=args.text_model_name, trainable=args.text_trainable)

        self.image_projection = ProjectionHead(
            embedding_dim=args.image_embedding, projection_dim=args.projection_dim, dropout=args.dropout)

        self.sentences_projection = ProjectionHead(
            embedding_dim=args.text_embedding, projection_dim=args.projection_dim, dropout=args.dropout)

        self.graph_projection = ProjectionHead(
            embedding_dim=args.graph_embedding, projection_dim=args.projection_dim, dropout=args.dropout)

        self.post_process_layers = nn.Sequential(
            nn.BatchNorm1d(256, affine=False),
            nn.Dropout(p=args.dropout),
            nn.Linear(256, 2),
            nn.GELU()
        )
                        
    def forward_graph(self, data, sentences_feature):
        # Getting graph embedding
        graph_features = self.graph_encoder(data, sentences_feature)
        graph_extract = self.post_process_layers(graph_features)        
        graph_extract = F.log_softmax(graph_extract, dim=-1)        
        return graph_extract

    def get_image_aggregation(self, image):
        x = self.image_encoder(image)
        x = self.image_projection(x)
        return x


    def forward(self, data):
        # IMAGE
        image_aggregation = self.get_image_aggregation(data.pills_images)
        # TEXT
        sentences_feature = self.sentences_encoder(data.text_sentences_ids, data.text_sentences_mask)
        sentences_projection = self.sentences_projection(sentences_feature)

        # GRAPH
        graph_extract = self.forward_graph(data, sentences_feature)
        
        return image_aggregation, sentences_projection, graph_extract
