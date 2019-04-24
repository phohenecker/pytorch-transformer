pytorch-transformer
===================


This repository provides a PyTorch implementation of the *Transformer* model that has been introduced in the paper
*Attention Is All You Need* (Vaswani et al. 2017).


Installation
------------

The easiest way to install this package is via pip:

```bash
pip install git+https://github.com/phohenecker/pytorch-transformer
```


Usage
-----

```python
import transformer
model = transformer.Transformer(...)
```

##### 1. Computing Predictions given a Target Sequence

This is the default behaviour of a
[`Transformer`](src/main/python/transformer/transformer.py),
and is implemented in its
[`forward`](src/main/python/transformer/transformer.py#L205)
method:
```python
predictions = model(input_seq, target_seq)
```


##### 2. Evaluating the Probability of a Target Sequence

The probability of an output sequence given an input sequence under an already trained model can be evaluated by means
of the function
[`eval_probability`](src/main/python/transformer/transformer_tools.py#L46):
```python
probabilities = transformer.eval_probability(model, input_seq, target_seq, pad_index=...)
```

##### 3. Sampling an Output Sequence

Sampling a random output given an input sequence under the distribution computed by a model is realized by the function
[`sample_output`](src/main/python/transformer/transformer_tools.py#L115):

```python
output_seq = transformer.sample_output(model, input_seq, eos_index, pad_index, max_len)
```


Pretraining Encoders with BERT
------------------------------

For pretraining the encoder part of the transformer
(i.e.,[`transformer.Encoder`](src/main/python/transformer/encoder.py))
with BERT (Devlin et al., 2018), the class [`MLMLoss`](src/main/python/transformer/bert/mlm_loss.py) provides an
implementation of the masked language-model loss function.
A full example of how to implement pretraining with BERT can be found in
[`examples/bert_pretraining.py`](examples/bert_pretraining.py).


References
----------

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., Polosukhin, I. (2017).
> Attention Is All You Need.  
> Preprint at http://arxiv.org/abs/1706.03762.

> Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018).  
> BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.  
> Preprint at http://arxiv.org/abs/1810.04805.
