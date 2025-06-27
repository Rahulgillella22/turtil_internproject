import torch
from torch import nn
from transformers import BertModel, BertTokenizer

class TagPredictionModel(nn.Module):
    def __init__(self, num_labels):
        super(TagPredictionModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.activation = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        x = self.classifier(x)
        return self.activation(x)
