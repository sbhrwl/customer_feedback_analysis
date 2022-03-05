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
