# -*- coding: utf-8 -*-


import torch

from torch import nn

from transformer import enc_dec_base
from transformer import feed_forward_layer as ffl
from transformer import multi_head_attention as mha
from transformer import normalization


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
__date__ = "Aug 21, 2018"
__maintainer__ = "Patrick Hohenecker"
__email__ = "mail@paho.at"
__status__ = "Development"


# ==================================================================================================================== #
#  CLASS  E N C O D E R                                                                                                #
# ==================================================================================================================== #


class Encoder(nn.Module, enc_dec_base.EncDecBase):
    """The encoder that is used in the Transformer model."""
    
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        enc_dec_base.EncDecBase.__init__(self, *args, **kwargs)
        
        self._layers = nn.ModuleList([_EncoderLayer(self) for _ in range(self._num_layers)])
    
    #  METHODS  ########################################################################################################
    
    def forward(self, sequence: torch.FloatTensor, padding_mask: torch.ByteTensor=None) -> torch.FloatTensor:
        """Runs the encoder.
        
        Args:
            sequence (torch.FloatTensor): The input sequence as (batch-size x seq-len x dim-model)-tensor.
            padding_mask (torch.ByteTensor, optional):  Optionally, a padding mask as
                (batch-size x in-seq-len x in-seq-len)-tensor. To that end, ``1``s indicate those positions that are
                part of the according sequence, and ``0``s mark padding tokens.
        
        Returns:
            FloatTensor: The encoded sequence as (batch_size x seq_len x dim_model)-tensor.
        """
        assert sequence.dim() == 3
        assert sequence.size(2) == self._dim_model
        
        # apply all layers to the input
        for layer in self._layers:
            sequence = layer(sequence, padding_mask)
        
        # provide the final sequence
        return sequence
    
    def reset_parameters(self) -> None:
        for l in self._layers:
            l.reset_parameters()


# ==================================================================================================================== #
#  CLASS  _ E N C O D E R  L A Y E R                                                                                   #
# ==================================================================================================================== #


class _EncoderLayer(nn.Module):
    """One layer of the encoder.
    
    Attributes:
        attn: (:class:`mha.MultiHeadAttention`): The attention mechanism that is used to read the input sequence.
        feed_forward (:class:`ffl.FeedForwardLayer`): The feed-forward layer on top of the attention mechanism.
    """
    
    def __init__(self, parent: Encoder):
        """Creates a new instance of ``_EncoderLayer``.
        
        Args:
            parent (Encoder): The encoder that the layers is created for.
        """
        super().__init__()
        self.attn = mha.MultiHeadAttention(
                parent.num_heads,
                parent.dim_model,
                parent.dim_keys,
                parent.dim_values,
                parent.attention_dropout
        )
        self.feed_forward = ffl.FeedForwardLayer(parent.dim_model)
        self.norm = normalization.Normalization()
        self.dropout = nn.Dropout(parent.residual_dropout)

    #  METHODS  ########################################################################################################
    
    def forward(self, sequence: torch.FloatTensor, padding_mask: torch.ByteTensor) -> torch.FloatTensor:
        """Runs the layer.
        
        Args:
            sequence (torch.FloatTensor): The input sequence as (batch_size x seq_len x dim_model)-tensor.
            padding_mask (torch.ByteTensor): The padding mask as (batch_size x seq_len x seq_len)-tensor or ``None`` if
                no mask is used.
        
        Returns:
            torch.FloatTensor: The encoded sequence as (batch_size x seq_len x dim_model)-tensor.
        """
        # compute attention sub-layer
        sequence = self.norm(self.dropout(self.attn(sequence, sequence, sequence, mask=padding_mask)) + sequence)
        
        # compute feed-forward sub-layer
        sequence = self.norm(self.dropout(self.feed_forward(sequence)) + sequence)
        
        return sequence
    
    def reset_parameters(self) -> None:
        """Resets all trainable parameters of the module."""
        self.attn.reset_parameters()
        self.feed_forward.reset_parameters()
