# -*- coding: utf-8 -*-

"""This module provides various utility functions."""


import itertools
import numbers

import numpy as np
import torch

from torch import nn


__author__ = "Patrick Hohenecker"
__copyright__ = (
        "Copyright (c) 2018, Patrick Hohenecker\n"
        "All rights reserved.\n"
        "\n"
        "Redistribution and use in source and binary forms, with or without\n"
        "modification, are permitted provided that the following conditions are met:\n"
        "\n"
        "1. Redistributions of source code must retain the above copyright notice, this\n"
        "   list of conditions and the following disclaimer.\n"
        "2. Redistributions in binary form must reproduce the above copyright notice,\n"
        "   this list of conditions and the following disclaimer in the documentation\n"
        "   and/or other materials provided with the distribution.\n"
        "\n"
        "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND\n"
        "ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED\n"
        "WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE\n"
        "DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR\n"
        "ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES\n"
        "(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;\n"
        "LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND\n"
        "ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT\n"
        "(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS\n"
        "SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
)
__license__ = "BSD-2-Clause"
__version__ = "2018.1"
__date__ = "Aug 29, 2018"
__maintainer__ = "Patrick Hohenecker"
__email__ = "mail@paho.at"
__status__ = "Development"


def create_padding_mask(seq: torch.LongTensor, pad_index: int) -> torch.ByteTensor:
    """Creates a mask for the provided sequence that indicates which of the tokens are actual data and which are just
    padding.
    
    Args:
        seq (torch.LongTensor): The input sequences that the padding mask is created for. ``seq`` has to be a
            ``LongTensor`` of shape (batch-size x seq-len).
        pad_index (int): The index that indicates a padding token.
    
    Returns:
        torch.ByteTensor: A binary mask where ``1``s represent tokens that belong to the actual sequences and ``0``s
            indicate padding. The provided mask has the shape (batch-len x seq-len x seq-len).
    """
    # sanitize args
    if not isinstance(seq, torch.LongTensor) and not isinstance(seq, torch.cuda.LongTensor):
        raise TypeError("<seq> has to be a LongTensor!")
    if seq.dim() != 2:
        raise ValueError("<seq> has to be a 2-dimensional tensor!")
    if not isinstance(pad_index, int):
        raise TypeError("<pad_index> has to be an int!")
    
    seq_len = seq.size(1)
    
    return (seq != pad_index).unsqueeze(1).expand(-1, seq_len, -1)


def create_positional_emb(max_seq_len: int, embedding_size: int, dim_model: int) -> nn.Embedding:
    """Creates positional embeddings.

    Args:
        max_seq_len (int): The maximum length of any input sequence, which corresponds with the total number of
            embedding vectors needed.
        embedding_size (int): The size of the embeddings to create.
        dim_model (int): The default layer size used in the model.

    Returns:
        nn.Embedding: The created positional embeddings.
    """
    emb_matrix = (
            [
                    np.sin(np.array(range(max_seq_len), dtype=np.float32) / (10000 ** (i / dim_model))),
                    np.cos(np.array(range(max_seq_len), dtype=np.float32) / (10000 ** (i / dim_model)))
            ]
            for i in range(0, embedding_size, 2)
    )
    emb_matrix = np.stack(itertools.chain(*emb_matrix)).T

    # if max_seq_len is an odd number, than the last entry of the embedding matrix has to be removed again
    if emb_matrix.shape[0] > max_seq_len:
        emb_matrix = emb_matrix[:-1]

    return nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix))


def create_shifted_output_mask(seq: torch.Tensor) -> torch.ByteTensor:
    """Creates a mask that prevents the decoder to attend future outputs.
    
    For each sample in the provided batch, the created mask is a square matrix that contains one row for every
    position in the output sequence. Each of these rows indicates those parts of the sequence that may be considered in
    order to compute the respective output, i.e., those output values that have been computed earlier.
    
    Args:
        seq (torch.Tensor): The output sequence that the padding is mask is created for. ``seq`` has to be a tensor of
            shape (batch-size x seq-len x ...), i.e., it has to have at least two dimensions.
    
    Returns:
        torch.ByteTensor: A binary mask where ``1``s represent tokens that should be considered for the respective
            position and ``0``s indicate future outputs. The provided mask has shape (batch-size x seq-len x seq-len).
    """
    # sanitize args
    if not isinstance(seq, torch.Tensor):
        raise TypeError("<seq> has to be a Tensor!")
    if seq.dim() < 2:
        raise ValueError("<seq> has to be at least a 2-dimensional tensor!")
    
    batch_size = seq.size(0)
    seq_len = seq.size(1)
    
    # create a mask for one sample
    mask = 1 - seq.new(seq_len, seq_len).fill_(1).triu(diagonal=1).byte()
    
    # copy the mask for all samples in the batch
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    return mask


def shift_output_sequence(seq: torch.Tensor, zero_range: numbers.Real=1e-22) -> torch.Tensor:
    """Shifts the provided output sequence one position to the right.
    
    To shift the sequence, this function truncates the last element of and prepends a zero-entry to every sample of
    the provided batch. However, to prevent ``nan`` values in the gradients of tensors created by means of
    ``torch.std``, the prepended tensors are not actually set to 0, but sampled uniformly from a tiny interval around 0,
    which may be adjusted via the arg ``zero_range``.
    
    Args:
        seq (torch.Tensor): The sequence to shift as (batch-size x seq-length x dim-model)-tensor.
        zero_range (numbers.Real, optional): Specifies the range to sample zero-entries from as closed interval
            [``zero_range``, ``-zero_range``].
    
    Returns:
        torch.Tensor: The shifted sequence, which, just like ``seq``, is a (batch-size x seq-length x dim-model)-tensor.
    """
    # sanitize args
    if not isinstance(seq, torch.Tensor):
        raise TypeError("<seq> has to be a tensor!")
    if seq.dim() != 3:
        raise ValueError("Expected <seq> to be 3D, but {} dimensions were encountered!".format(seq.dim()))
    if not isinstance(zero_range, numbers.Real):
        raise TypeError("The <zero_range> has to be a real number!")
    zero_range = float(zero_range)
    if zero_range <= 0:
        raise ValueError("The <zero_range> has to be a positive number!")
    
    return torch.cat(
            [
                    seq.new(seq.size(0), 1, seq.size(2)).uniform_(-zero_range, zero_range),
                    seq[:, :-1, :]
            ],
            dim=1
    )
