from tqdm import tqdm
import csv
import torch
from torch.utils.data import Dataset
from config import (
  TAGS_NUM,
  TAGS_PUNCT,
  unk_id,
  eos_id,
  pad_id,
  sos_id,
  tokenizer,
)


class TextDataset(Dataset):
  def __init__(self, text_file, sequence_len=256):
    lines = []

    with open(text_file) as infile:
      for token, tag_punct, tag_num in csv.reader(infile, delimiter="\t"):
        lines.append((token, TAGS_PUNCT.index(tag_punct), TAGS_NUM.index(tag_num)))

    idx = 0
    self.items = []
    with tqdm(total=len(lines), disable=True) as pbar:
      while idx < len(lines):
        x = [sos_id]
        y = [[0, 0]]
        y_mask = [1]

        while len(x) < sequence_len - 1 and idx < len(lines):
          word, tag_punct_id, tag_num_id = lines[idx]
          tokens = tokenizer.tokenize(word)

          if len(tokens) + len(x) >= sequence_len:
            break
          else:
            for i in range(len(tokens) - 1):
              x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
              y.append([0, 0])
              y_mask.append(0)
            if len(tokens) > 0:
              x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
            else:
              x.append(unk_id)
            y.append([tag_punct_id, tag_num_id])
            y_mask.append(1)
            idx += 1
            pbar.update()

        x.append(eos_id)
        y.append([0, 0])
        y_mask.append(1)

        if len(x) < sequence_len:
          x = x + [pad_id for _ in range(sequence_len - len(x))]
          y = y + [[0, 0] for _ in range(sequence_len - len(y))]
          y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]

        attn_mask = [1 if token != pad_id else 0 for token in x]
        self.items.append([x, y, attn_mask, y_mask])

  def __len__(self):
    return len(self.items)

  def __getitem__(self, i):
    x, y, attn_mask, y_mask = self.items[i]
    x = torch.tensor(x)

    y_punct = torch.tensor([item[0] for item in y])
    y_num = torch.tensor([item[1] for item in y])

    attn_mask = torch.tensor(attn_mask)
    y_mask = torch.tensor(y_mask)
    return x, y_punct, y_num, attn_mask, y_mask


if __name__ == "__main__":
  ds = TextDataset(text_file="data/test.txt")
  print(len(ds))
  print(ds[0])
