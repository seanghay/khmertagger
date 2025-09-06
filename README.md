# KhmerTagger: Inverse Text Normalization for Khmer Automatic Speech Recognition

Keyword: Khmer language, Speech recognition, Inverse text normalization

Khmer Automatic Speech Recognition (ASR) systems often produce raw text that lacks
punctuation, spaces, and numbers, making it less readable. We propose a method to improve
the readability of these outputs by utilizing XLM-ROBERTa, a multilingual transformer model
trained on 100 languages, including Khmer. Our approach enhances an existing punctuation
restoration framework by adding a number recognition component. Trained on Khmer news
data, the model transforms raw ASR outputs into more polished, reader-friendly text. The code
is publicly available for further research and development.

### Introduction

The Khmer language, like Thai, Lao, and Japanese, does not use explicit word boundaries as
seen in languages like English or French. Instead, readers rely on contextual cues to recognize
word separations.
In Khmer, spaces serve a unique purpose. While they do not indicate word boundaries, their
placement carries various meanings, depending on context. Spaces are primarily used to mark
pauses, define boundaries, and enhance readability.
When working with text output from ASR (Automatic Speech Recognition) models, the text is
typically in a normalized form. Therefore, to make it suitable for practical use, it's necessary to
restore elements like spaces, punctuation, numbers, and symbols.

### Proposed method

In this work, we employ word boundary detection using khmercut, a tool designed for
segmentation of Khmer text into word units to make it ready to feed into the model.

Our model architecture is based on XLM-ROBERTA, a multilingual transformer model that serves
as the encoder. On top of this, we integrate a bidirectional Long Short-Term Memory (BILSTM)
network to capture contextual dependencies. Following the BILSTM, we apply a single fully
connected linear layer. The final output comprises two independent classification heads.

We extend the work of xashru's punctuation restoration model by introducing an additional
classification head for number recognition. This results in a model capable of handling two types
of outputs: punctuation (including space, comma, question mark, exclamation mark, and other
symbols) and numeric entities.

A post-processing step is employed after classification, where predicted tokens are converted
back into their corresponding numbers or symbols, ensuring proper representation of the original
text. This enhances the accuracy of both punctuation restoration and numeric entity recognition.

<img width="350" alt="Untitled Diagram-Page-2 drawio" src="https://github.com/user-attachments/assets/120b92b2-3140-4c84-b37c-26814ee76203" />

### Experiment


The model was trained on publicly available Khmer news data sourced from a variety of
websites, consisting of a total of 1.5 million tokens. This dataset includes diverse linguistic
contexts, contributing to the robustness of the model in handling punctuation and number recognition tasks in the Khmer language.

The training process yielded an accuracy of 97.2%, demonstrating the effectiveness of the
architecture in accurately predicting both punctuation and numeric entities in Khmer text.


```bibtex
@misc{khmertagger2025,
  author = {Seanghay Yath},
  title = {KhmerTagger: Inverse Text Normalization for Khmer Automatic Speech Recognition},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/seanghay/khmertagger}},
  note = {Open source project for Khmer punctuation restoration and number recognition using XLM-ROBERTa}
}
```
