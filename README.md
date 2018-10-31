pytorch-transformer
===================


This repository provides a PyTorch implementation of the *Transformer* model that has been introduced in the paper
*Attention Is All You Need* (Vaswani et al. 2017).


Installation
------------

The easiest way to install this package is via pip:

```bash
pip install git+https://git.paho.at/phohenecker/pytorch-transformer
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


References
----------

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., Polosukhin, I.  
> Attention Is All You Need.  
> Preprint at http://arxiv.org/abs/1706.03762, 2017.
