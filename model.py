import torch.nn as nn
import torch
from transformers import XLMRobertaModel
from config import MODEL_ID


class KhmerTagger(nn.Module):
  def __init__(self, n_punct_features, n_num_features, n_dim=768, n_hidden=1024):
    super(KhmerTagger, self).__init__()
    self.encoder = XLMRobertaModel.from_pretrained(MODEL_ID)
    self.lstm = nn.LSTM(
      n_dim, n_hidden, num_layers=1, bidirectional=True, batch_first=True
    )
    self.fc = nn.Linear(n_hidden * 2, n_hidden)
    self.fc_punct = nn.Linear(n_hidden, n_punct_features)
    self.fc_num = nn.Linear(n_hidden, n_num_features)

  def forward(self, x, attention_mask):
    x = self.encoder(x, attention_mask=attention_mask)[0]
    x = torch.transpose(x, 0, 1)
    x, (_, _) = self.lstm(x)
    x = torch.transpose(x, 0, 1)
    x = self.fc(x)
    return self.fc_punct(x), self.fc_num(x)


if __name__ == "__main__":
  x = torch.randint(0, 1000, (1, 100))
  mask = torch.ones((1, 100))
  model = KhmerTagger(10, 20)

  logits_punct, logits_num = model(x, attention_mask=mask)
  assert logits_punct.shape[1] == 100
  assert logits_num.shape[1] == 100
  assert logits_punct.shape[2] == 10
  assert logits_num.shape[2] == 20
