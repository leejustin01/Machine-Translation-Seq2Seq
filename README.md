# Machine Translation with Seq2Seq and Attention

A neural machine translation model that translates text using a sequence-to-sequence (Seq2Seq) architecture with an attention mechanism. The model is trained on the Multi30k dataset to translate sentences between languages.

## Overview

This project implements a Seq2Seq model with attention for machine translation. An encoder processes the input sentence into hidden representations, while a decoder generates the translated output word-by-word, using attention to focus on relevant parts of the input sequence.

## Features

- Sequence-to-sequence architecture for translation  
- Attention mechanism for improved context handling  
- Encoder–decoder model for variable-length sequences  
- Training and evaluation on the Multi30k dataset  

## Example

```python
train_loader, val_loader, test_loader, vocabs = getMulti30kDataloadersAndVocabs(config["bs"])
model = Seq2Seq(len(vocabs["de"]), len(vocabs["en"]), config["embed_dim"], 
                config["enc_dim"], config["dec_dim"], config["kq_dim"], config["attn"], config["dropout"])
torch.compile(model)
train(model, train_loader, val_loader, vocabs)
```

Acknowledgement

This project was developed as part of a university assignment for **CS 435 - Applied Deep Learning** at Oregon State University.
