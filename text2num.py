scales = [
  (2, "រយ"),
  (3, "ពាន់"),
  (4, "ម៉ឺន"),
  (5, "សែន"),
  (6, "លាន"),
  (9, "ប៊ីលាន"),
  (12, "ទ្រីលាន"),
  (15, "ក្វាទ្រីលាន"),
  (18, "គ្វីនទីលាន"),
  (21, "សិចទីលាន"),
  (24, "សិបទីលាន"),
  (27, "អុកទីលាន"),
  (30, "ណូនីលាន"),
  (33, "ដេស៊ីលាន"),
  (36, "អាន់ដេស៊ីលាន"),
]

scales = {v: 10**k for k, v in scales}

number_words = {
  "សូន្យ": 0,
  "មួយ": 1,
  "ពីរ": 2,
  "ពី": 2,
  "បី": 3,
  "បួន": 4,
  "ប្រាំ": 5,
  "ប្រាំមួយ": 6,
  "ប្រាំពីរ": 7,
  "ប្រាំពី": 7,
  "ប្រាំបី": 8,
  "ប្រាំបួន": 9,
  "ដប់": 10,
  "ម្ភៃ": 20,
  "សាមសិប": 30,
  "សាម": 30,
  "សែសិប": 40,
  "សែ": 40,
  "ហាសិប": 50,
  "ហា": 50,
  "ហុកសិប": 60,
  "ហុក": 60,
  "ចិតសិប": 70,
  "ចិត": 70,
  "ប៉ែតសិប": 80,
  "ប៉ែត": 80,
  "កៅសិប": 90,
  "កៅ": 90,
  "ប្រាំរយ": 500,
}

for scale, value in scales.items():
  number_words[scale] = value


def text2num(words):
  total = 0
  current = 0
  scale = 0

  incomplete = False
  tokens = []

  for w in words:
    if w not in number_words:
      if incomplete:
        tokens[-1] += w
        continue

      incomplete = True
      tokens.append(w)
      continue

    if incomplete and w == "លាន":
      tokens[-1] += w
      incomplete = False
      continue

    incomplete = False
    tokens.append(w)

  for word in tokens:
    if word in number_words:
      if word == "រយ":
        current *= 100
      elif word in scales:
        scale = scales[word]
        total += current * scale
        current = 0
        scale = 0
      else:
        current += number_words[word]

  return total + (current * (100**scale))