# BERT
Bidirectional Encoder Representations from Transformers
- [Paper](https://arxiv.org/pdf/1810.04805.pdf)
- [Blog1](https://huggingface.co/blog/bert-101)
- [Blog2](https://jalammar.github.io/illustrated-bert/)

## [Example](https://albertauyeung.github.io/2020/06/19/bert-tokenization.html/)
```python
# Import tokenizer from transformers package
from transformers import BertTokenizer

# Load the tokenizer of the "bert-base-cased" pretrained model
# See https://huggingface.co/transformers/pretrained_models.html for other models
tz = BertTokenizer.from_pretrained("bert-base-cased")

# The senetence to be encoded
sent = "Let's learn deep learning!"

# Encode the sentence
encoded = tz.encode_plus(
    text=sent,  # the sentence to be encoded
    add_special_tokens=True,  # Add [CLS] and [SEP]
    max_length = 64,  # maximum length of a sentence
    pad_to_max_length=True,  # Add [PAD]s
    return_attention_mask = True,  # Generate the attention mask
    return_tensors = 'pt',  # ask the function to return PyTorch tensors
)

# Get the input IDs and attention mask in tensor format
input_ids = encoded['input_ids']
attn_mask = encoded['attention_mask']
```
### Preparing a sentence for input to the BERT model
- For simplicity, we assume the maximum length is 10 in the example below (while in the original model it is set to be 512).
```
# Original Sentence
Let's learn deep learning!

# Tokenized Sentence
['Let', "'", 's', 'learn', 'deep', 'learning', '!']

# Adding [CLS] and [SEP] Tokens
['[CLS]', 'Let', "'", 's', 'learn', 'deep', 'learning', '!', '[SEP]']

# Padding
['[CLS]', 'Let', "'", 's', 'learn', 'deep', 'learning', '!', '[SEP]', '[PAD]']

# Converting to IDs
[101, 2421, 112, 188, 3858, 1996, 3776, 106, 102, 0]
```
## BERT: From Decoders to Encoders
- Masked Language Model
- Two-sentence Tasks
- Task specific-Models
- BERT for feature extraction
## [BERT Colab](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
- Trains on TPU
