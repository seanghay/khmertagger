import regex as re
import csv
from tqdm import tqdm
from khmercut import tokenize
from multiprocessing import Pool
import tha.decimals

re_num = re.compile(r"[\d\u17e0-\u17e9]+")
re_ending_punct = re.compile(r"[។៕?!]+$")
re_html_comments = re.compile(r"<!--(.*?)-->")
re_iter = re.compile(
  r"([?! ។៕]+)|([0-9\u17e0-\u17e9,\.]+)|([a-z'A-Z]+)|([^?! ។៕0-9\u17e0-\u17e9,\.a-z'A-Z]+)"
)


def transform(item):
  if item is None:
    return

  if item["title"] is None:
    return

  if item["content"] is None:
    return

  title = item["title"].replace("\n", " ").strip()
  content = item["content"].strip()
  content = re_html_comments.sub("", content).strip().replace("\n", " ")

  if not re_ending_punct.search(title):
    title += "។"

  text = title + content
  values = []

  for m in re_iter.finditer(text):
    if m[3]:
      values.append((m[3], "0", "0"))
      continue

    if m[4]:
      for word in tokenize(m[4]):
        values.append((word, "0", "0"))
      continue

    if m[1]:
      tag = m[1].strip()
      tag = tag[-1] if tag else "SPACE"
      tag = "។" if tag == "៕" else tag

      if len(values) > 0:
        if values[-1][1] == "0":
          values[-1] = (values[-1][0], tag, values[-1][2])

      continue

    if m[2] and re_num.search(m[2]):
      s = tha.decimals.processor(m[2].strip(",").strip(".")).replace("▁", "")
      for idx, t in enumerate(tokenize(s)):
        values.append((t, "0", "NUMBER_B" if idx == 0 else "NUMBER_I"))
      continue

  return values


if __name__ == "__main__":
  with open("data/koh.csv") as infile:
    reader = csv.DictReader(infile)
    items = []
    for item in reader:
      if len(items) > 24000:
        break
      items.append(item)

  with open("data/train.txt", "w") as outfile:
    data = items[0:20000]
    writer = csv.writer(outfile, delimiter="\t")
    with tqdm(total=len(data)) as pbar:
      with Pool() as pool:
        for values in pool.imap_unordered(transform, data):
          if values is not None:
            for value in values:
              writer.writerow(value)
          pbar.update()

  with open("data/dev.txt", "w") as outfile:
    data = items[20000:]
    writer = csv.writer(outfile, delimiter="\t")
    with tqdm(total=len(data)) as pbar:
      with Pool() as pool:
        for values in pool.imap_unordered(transform, data):
          if values is not None:
            for value in values:
              writer.writerow(value)
          pbar.update()
