import torch
from model import KhmerTagger
from config import TAGS_PUNCT, TAGS_NUM, tokenizer, eos_id, sos_id, pad_id, unk_id
from khmercut import tokenize
from text2num import text2num

digits_translation = str.maketrans(
  {
    "0": "០",
    "1": "១",
    "2": "២",
    "3": "៣",
    "4": "៤",
    "5": "៥",
    "6": "៦",
    "7": "៧",
    "8": "៨",
    "9": "៩",
  }
)


def punctuate(model, words, device):
  sequence_len = 256
  idx = 0
  decode_idx = 0

  while idx < len(words):
    x = [sos_id]
    y_mask = [0]

    while len(x) < sequence_len - 1 and idx < len(words):
      word = words[idx]
      tokens = tokenizer.tokenize(word)

      if len(tokens) + len(x) >= sequence_len:
        break
      else:
        for i in range(len(tokens) - 1):
          x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
          y_mask.append(0)
        if len(tokens) > 0:
          x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
        else:
          x.append(unk_id)
        y_mask.append(1)
        idx += 1

    x.append(eos_id)
    y_mask.append(0)

    if len(x) < sequence_len:
      x = x + [pad_id for _ in range(sequence_len - len(x))]
      y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]

    attn_mask = [1 if token != pad_id else 0 for token in x]

    x = torch.tensor(x).to(device).reshape(1, -1)
    attn_mask = torch.tensor(attn_mask).reshape(1, -1).to(device)
    y_punct, y_num = model(x, attn_mask)

    y_punct = y_punct.view(-1, y_punct.shape[2])
    y_num = y_num.view(-1, y_num.shape[2])

    y_punct = torch.argmax(y_punct, dim=1).view(-1).cpu().numpy()
    y_num = torch.argmax(y_num, dim=1).view(-1).cpu().numpy()

    for i in range(len(y_mask)):
      if y_mask[i] == 1:
        yield (words[decode_idx], TAGS_PUNCT[y_punct[i]], TAGS_NUM[y_num[i]])
        decode_idx += 1


if __name__ == "__main__":
  device = "cuda"
  checkpoint_path = "logs_take/checkpoint-30.pth"

  model = KhmerTagger(n_punct_features=len(TAGS_PUNCT), n_num_features=len(TAGS_NUM))
  model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
  model.to(device)

  with open("data/sample.txt") as infile:
    tokens = [token for line in infile for token in line.rstrip("\n").split()]
    text = "".join(tokens)
    tokens = tokenize(text)

    pos = 0
    outputs = list(punctuate(model, tokens, device))
    results = []

    while pos < len(outputs):
      token, tag_punct, tag_num = outputs[pos]
      if tag_num == "NUMBER_B":
        ss = pos
        pos += 1
        while pos < len(outputs):
          tag = outputs[pos][2]
          if tag == "NUMBER_B" or tag == "0":
            break
          pos += 1

        targets = outputs[ss:pos]
        punct = targets[-1][1]
        words = [target[0] for target in targets]
        num = text2num(words)

        results.append((num, punct))
        continue

      results.append((token, tag_punct))
      pos += 1

    # merge
    tag2text = {
      "0": "",
      "SPACE": " ",
      "។": "។\n\n",
      "?": "? ",
      "!": "! ",
    }

    text_punct = ""
    idx = 0

    units = {
      "ភាគរយ": "%",
      "គីឡូបៃ": "KB",
      "មេកាបៃ": "MB",
      "ជីហ្គាបៃ": "GB",
      "ប៉េតាបៃ": "PB",
    }

    for token, tag in results:
      if isinstance(token, int):
        token = str(token).translate(digits_translation)

        # two numbers siting next to each other
        if idx > 0 and isinstance(results[idx - 1][0], int):
          if tag == "0":
            tag = "SPACE"

      if idx > 0 and isinstance(results[idx - 1][0], int) and token in units:
        token = units[token]

        if tag == "0":
          tag = "SPACE"

      text_punct += token + tag2text[tag]
      idx += 1

    with open("data/sample.out.txt", "w") as outfile:
      outfile.write(text_punct.strip())
