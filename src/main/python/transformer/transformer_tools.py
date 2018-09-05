# -*- coding: utf-8 -*-

"""This module contains utility functions for working with the Transformer model."""


import numpy as np
import torch

from transformer import transformer
from transformer import util


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
__date__ = "Aug 30, 2018"
__maintainer__ = "Patrick Hohenecker"
__email__ = "mail@paho.at"
__status__ = "Development"


def eval_probability(
        model: transformer.Transformer,
        input_seq: torch.LongTensor,
        target_seq: torch.LongTensor,
        pad_index: int=None
) -> torch.FloatTensor:
    """Computes the probability that the provided model computes a target sequence given an input sequence.
    
    Args:
         model (:class:`transformer.Transformer`): The model to use.
         input_seq (torch.LongTensor): The input sequence to be provided to the model. This has to be a
            (batch-size x input-seq-len)-tensor.
         target_seq (torch.LongTensor): The target sequence whose probability is being evaluated. This has to be a
            (batch-size x target-seq-len)-tensor.
         pad_index (int, optional): The index that indicates a padding token in a sequence. If ``target_seq`` is padded,
            then the ``pad_index`` has to be provided in order to allow for computing the probabilities for relevant
            parts of the target sequence only.
    
    Returns:
        torch.FloatTensor: A 1D-tensor of size (batch-size), which contains one probability for each sample in
            ``input_seq`` and ``target_seq``, respectively.
    """
    if not isinstance(model, transformer.Transformer):
        raise TypeError("The <model> has to be a transformer.Transformer!")
    if not isinstance(input_seq, torch.LongTensor) and not isinstance(input_seq, torch.cuda.LongTensor):
        raise TypeError("The <input_seq> has to be a LongTensor!")
    if input_seq.dim() != 2:
        raise ValueError("<input_seq> has to be a 2D-tensor!")
    if input_seq.is_cuda:
        if not isinstance(target_seq, torch.cuda.LongTensor):
            raise TypeError("The <target_seq> has to be of the same type as <input_seq>, i.e., cuda.LongTensor!")
    elif not isinstance(target_seq, torch.LongTensor):
        raise TypeError("The <target_seq> has to be of the same type as <input_seq>, i.e., LongTensor!")
    if target_seq.dim() != 2:
        raise ValueError("<input_seq> has to be a 2D-tensor!")
    if input_seq.size(0) != target_seq.size(0):
        raise ValueError("<input_seq> and <target_seq> use different batch sizes!")
    if pad_index is not None and not isinstance(pad_index, int):
        raise TypeError("The <pad_index>, if provided, has to be an integer!")
    
    batch_size = input_seq.size(0)
    max_seq_len = input_seq.size(1)
    
    # run the model to compute the needed probabilities
    predictions = model(input_seq, target_seq)
    
    # determine the lengths of the target sequences
    if pad_index is not None:
        mask = util.create_padding_mask(target_seq, pad_index)[:, 0, :]
        seq_len = mask.sum(dim=1).cpu().numpy().tolist()
    else:
        seq_len = (np.ones(batch_size, dtype=np.long) * max_seq_len).tolist()
    
    # compute the probabilities for each of the provided samples
    sample_probs = torch.ones(batch_size)
    for sample_idx in range(batch_size):  # iterate over each sample
        for token_idx in range(seq_len[sample_idx]):  # iterate over each position in the output sequence
            sample_probs[sample_idx] *= predictions[sample_idx, token_idx, target_seq[sample_idx, token_idx]].item()
    
    return sample_probs
