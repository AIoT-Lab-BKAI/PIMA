import torch
from torch import nn
from torchvision import models
import config as CFG
import timm
from torch_geometric.nn import SAGEConv
from transformers import BertModel, AutoModel
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


class BERTxSAGE(torch.nn.Module):
    def __init__(self, n_classes=len(CFG.LABELS), hidden_size=768, dropout_rate=0.2, bert_model=CFG.text_encoder_model):
        super().__init__()

        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.BERT = BertModel.from_pretrained(bert_model)
        self.hidden_size = hidden_size
        self.dense = nn.Linear(self.hidden_size, self.hidden_size) 

        self.activation = nn.Tanh()
        self.conv1 = SAGEConv(self.hidden_size + 2, 512)
        self.conv2 = SAGEConv(512,  256)

        for param in self.BERT.parameters():
            param.requires_grad = False

    def forward(self, data):
        # for transductive setting with full-batch update
        edge_index, edge_weight = data.edge_index, data.edge_attr
        bert_output = self.BERT(attention_mask=data.attention_mask,
                                input_ids=data.input_ids)
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py

        first_token_tensor = bert_output['last_hidden_state'][:, 0]
        pooled_output = self.dense(first_token_tensor)
        
        x = self.activation(pooled_output)
        x = torch.cat((x, data.p_num, data.text_len), dim=1)

        x = F.dropout(F.relu(self.conv1(x, edge_index, edge_weight)),
                      p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return x

class sentencesTransformer(torch.nn.Module):
    def __init__(self, model_name='sentence-transformers/paraphrase-mpnet-base-v2', trainable=CFG.text_trainable):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = trainable

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, text_sentences_ids, text_sentences_mask):
        model_output = self.model(text_sentences_ids, text_sentences_mask)
        return self.mean_pooling(model_output, text_sentences_mask)

class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.text_pretrained, trainable=CFG.text_trainable):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)
            
        for p in self.model.parameters():
            p.requires_grad = trainable
        
    def forward(self, x):
        # TODO: CHECK IT AGAIN 
        _, pooled_output = self.model(input_ids= x.input_ids, attention_mask=x.attention_mask, return_dict=False)
        return pooled_output

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """
    def __init__(self, model_name=CFG.image_model_name, pretrained=CFG.image_pretrained, trainable=CFG.image_trainable, image_num_classes=None):
        super().__init__()
        if model_name == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
            for param in self.model.parameters():
                param.requires_grad = trainable

        if image_num_classes is not None:
            self.model.fc = nn.Linear(self.model.fc.in_features, image_num_classes)
        else:
            self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)


class ImageEncoderTimm(nn.Module):
    def __init__(self, model_name=CFG.image_model_name, pretrained=CFG.image_pretrained, trainable=CFG.image_trainable):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for param in self.model.parameters():
            param.requires_grad = trainable
    def forward(self, x):
        return self.model(x)


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
