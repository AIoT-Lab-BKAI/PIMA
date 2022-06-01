from torch import nn
import torch.nn.functional as F
from models import ImageEncoder, ProjectionHead, ImageEncoderTimm, sentencesTransformer, SBERTxSAGE
import torch


class PrescriptionPill(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.image_encoder = ImageEncoderTimm(image_model_name=args.image_model_name,
                                              image_pretrained=args.image_pretrained, image_trainable=args.image_trainable)

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

        self.W1 = nn.Linear(args.image_embedding, args.projection_dim)
        self.W2 = nn.Linear(args.image_embedding, args.projection_dim)

    def forward_graph(self, data, sentences_feature):
        # Getting graph embedding
        graph_features = self.graph_encoder(data, sentences_feature)
        return graph_features

    def get_image_aggregation(self, image, label, label_batch=None):
        x = self.image_encoder(image) # (img_batch, img_embedding)

        if label_batch is None:
            calculate_mean = torch.zeros(x.shape[0], x.shape[1]).cuda()
            for idx, value in enumerate(label):
                if sum(label != value) == 0:
                    continue
                other_image = x[label != value].detach()
                calculate_mean[idx, :] = other_image.mean(dim=0)
            x = self.W1(x) + self.W2(calculate_mean)
            
        else:
            unique_label = []
            for value in label_batch:
                if value not in unique_label:
                    unique_label.append(value)
            
            calculate_mean = torch.tensor([]).cuda()
            for value in unique_label:
                x_new = x[label_batch == value].detach()
                label_new = label[label_batch == value]
                calculate_mean_new = torch.zeros(x_new.shape[0], x_new.shape[1]).cuda()
                for idx, value in enumerate(label_new):
                    if sum(label_new != value) == 0:
                        continue
                    other_image = x_new[label_new != value].detach()
                    calculate_mean_new[idx, :] = other_image.mean(dim=0)
                calculate_mean = torch.cat((calculate_mean, calculate_mean_new), dim=0)
            
            x = self.W1(x) + self.W2(calculate_mean)

        return x

    def get_image_aggregation_all(self, image):
        x = self.image_encoder(image)
        x = x.mean(dim=0, keepdim=True)
        x = self.image_projection(x)
        return x
    
    def get_image_aggregation_all_version_2(self, image, label, label_batch=None):
        x = self.image_encoder(image) # (img_batch, img_embedding)
        unique_label = []
                
        for value in label_batch:
            if value not in unique_label:
                unique_label.append(value)
        calculate_mean = torch.tensor([]).cuda()
        for value in unique_label: 
            x_new = x[label_batch == value].detach()
            label_new = label[label_batch == value]
            calculate_mean_new = torch.zeros(x_new.shape[0], x_new.shape[1]).cuda()
            for idx, value in enumerate(label_new):
                calculate_mean_new[idx, :] = x_new.mean(dim=0)
            calculate_mean = torch.cat((calculate_mean, calculate_mean_new), dim=0)
        
        x = self.image_projection(calculate_mean)
        return x 

    def forward(self, data):
        # IMAGE
        image_aggregation = self.get_image_aggregation(data.pills_images, data.pills_images_labels, data.pills_images_labels_idx)
                
        image_all_projection = self.get_image_aggregation_all_version_2(data.pills_images, data.pills_images_labels, data.pills_images_labels_idx)

        # TEXT
        sentences_feature = self.sentences_encoder(
            data.text_sentences_ids, data.text_sentences_mask)
        sentences_projection = self.sentences_projection(sentences_feature)

        # GRAPH
        graph_features = self.forward_graph(data, sentences_feature)
        graph_projection = self.graph_projection(graph_features)

        return image_aggregation, image_all_projection, sentences_projection, graph_projection
