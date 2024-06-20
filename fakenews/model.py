from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, bert_path, dropout, n_class):
        super(BertClassifier, self).__init__()
        self.bert_path = bert_path
        self.n_class = n_class
        self.dropout = nn.Dropout(dropout)
        self.bert = BertModel.from_pretrained(self.bert_path)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.n_class)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        out = self.dropout(out)
        out = self.fc(out)
        return out
