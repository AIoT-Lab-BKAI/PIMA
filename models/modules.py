import torch
from torch import nn
import timm
import config as CFG

from torch_geometric.nn import SAGEConv
from transformers import RobertaModel
from utils import LABELS
import torch.nn.functional as F
import random
import warnings
warnings.filterwarnings("ignore")

class BERTxSAGE(torch.nn.Module):
    def __init__(self, n_classes=len(LABELS), hidden_size=768, dropout_rate=0.2, bert_model="roberta-base"):
        super().__init__()

        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.BERT = RobertaModel.from_pretrained(bert_model)
        self.hidden_size = hidden_size
        self.dense1 = nn.Linear(self.hidden_size, self.hidden_size * 2)  # update
        self.activation = nn.Tanh()
        self.conv1 = SAGEConv(self.hidden_size + 2, 512)
        self.conv2 = SAGEConv(512,  256)
        # self.post_process_layers = nn.Sequential(
        #     nn.BatchNorm1d(256, affine= False),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(256, n_classes),
        #     nn.GELU()
        # )

        for param in self.BERT.parameters():
            param.requires_grad = False

    def forward(self, data):
        # print(data)
        # for transductive setting with full-batch update
        edge_index, edge_weight = data.edge_index, data.edge_attr
        bert_output = self.BERT(attention_mask=data.attention_mask,
                      input_ids=data.input_ids)
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py
        first_token_tensor = bert_output['last_hidden_state'][:, 0]
        pooled_output = self.dense1(first_token_tensor)
        x = self.activation(pooled_output)
        x = torch.add(torch.mul(x[:,self.hidden_size:],random.uniform(0,1)),x[:,:self.hidden_size]) #update
        x = torch.cat((x, data.p_num, data.text_len), dim=1)
        x = F.dropout(F.relu(self.conv1(x, edge_index, edge_weight)),
                    p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        # x = self.post_process_layers(x)

        # return F.log_softmax(x, dim=1)
        return x

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

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
