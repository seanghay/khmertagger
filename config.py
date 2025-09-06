from transformers import XLMRobertaTokenizer

MODEL_ID = "FacebookAI/xlm-roberta-base"
TAGS_PUNCT = ["0", "!", "?", "SPACE", "។"]
TAGS_NUM = ["0", "NUMBER_B", "NUMBER_I"]

unk_id = 3
eos_id = 2
pad_id = 1
sos_id = 0


tokenizer = XLMRobertaTokenizer.from_pretrained(
  MODEL_ID, clean_up_tokenization_spaces=False
)
