# -*- coding: utf-8 -*-


import torch

from torch import nn

from transformer import enc_dec_base
from transformer import feed_forward_layer as ffl
from transformer import multi_head_attention as mha
from transformer import normalization
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
__date__ = "Aug 21, 2018"
__maintainer__ = "Patrick Hohenecker"
__email__ = "mail@paho.at"
__status__ = "Development"


# ==================================================================================================================== #
#  CLASS  D E C O D E R                                                                                                #
# ==================================================================================================================== #


class Decoder(nn.Module, enc_dec_base.EncDecBase):
    """The decoder that is used in the Transformer model."""
    
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        enc_dec_base.EncDecBase.__init__(self, *args, **kwargs)

        self._layers = nn.ModuleList([_DecoderLayer(self) for _ in range(self._num_layers)])
    
    #  METHODS  ########################################################################################################
    
    def forward(
            self,
            in_sequence: torch.FloatTensor,
            out_sequence: torch.FloatTensor,
            padding_mask: torch.ByteTensor=None
    ) -> torch.FloatTensor:
        """Runs the decoder.

        Args:
            in_sequence (torch.FloatTensor): The input sequence as (batch-size x in-seq-len x dim_model)-tensor.
            out_sequence (torch.FloatTensor): The output sequence as (batch-size x out-seq-len x dim_model)-tensor.
            padding_mask (torch.ByteTensor, optional): Optionally, a padding mask as
                (batch-size x in-seq-len x in-seq-len)-tensor. To that end, ``1``s indicate those positions that are
                part of the according sequence, and ``0``s mark padding tokens.

        Returns:
            FloatTensor: The computed output as (batch_size x out-seq-len x dim_model)-tensor.
        """
        assert in_sequence.dim() == 3
        assert in_sequence.size(2) == self._dim_model
        assert out_sequence.dim() == 3
        assert out_sequence.size(0) == in_sequence.size(0)
        assert out_sequence.size(2) == self._dim_model
        if padding_mask is not None:
            assert padding_mask.dim() == 3
            assert padding_mask.size(0) == in_sequence.size(0)
            assert padding_mask.size(1) == in_sequence.size(1)
            assert padding_mask.size(2) == in_sequence.size(1)
        
        # create shifted output mask
        shifted_output_mask = util.create_shifted_output_mask(out_sequence)
        
        # shift provided target output to the right
        out_sequence = util.shift_output_sequence(out_sequence)
    
        # apply all layers to the input
        for layer in self._layers:
            out_sequence = layer(in_sequence, out_sequence, padding_mask, shifted_output_mask)
    
        # provide the created output
        return out_sequence
    
    def reset_parameters(self) -> None:
        for l in self._layers:
            l.reset_parameters()


# ==================================================================================================================== #
#  CLASS  _ D E C O D E R  L A Y E R                                                                                   #
# ==================================================================================================================== #


class _DecoderLayer(nn.Module):
    """One layer of the decoder.
    
    Attributes:
        attn_1 (:class:`mha.MultiHeadAttention`): The attention mechanism that is used to read from the output sequence.
        attn_2 (:class:`mha.MultiHeadAttention`): The encoder-decoder attention mechanism.
        feed_forward (:class:`ffl.FeedForwardLayer`): The feed-forward layer on top of the attention mechanisms.
    """
    
    def __init__(self, parent: Decoder):
        """Creates a new instance of ``_DecoderLayer``.

        Args:
            parent (Decoder): The decoder that the layers is created for.
        """
        super().__init__()
        self.attn_1 = mha.MultiHeadAttention(
                parent.num_heads,
                parent.dim_model,
                parent.dim_keys,
                parent.dim_values,
                parent.attention_dropout
        )
        self.attn_2 = mha.MultiHeadAttention(
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
    
    def forward(
            self,
            in_sequence: torch.FloatTensor,
            out_sequence: torch.FloatTensor,
            padding_mask: torch.ByteTensor,
            shifted_output_mask: torch.ByteTensor
    ) -> torch.FloatTensor:
        """Runs the layer.

        Args:
            in_sequence (torch.FloatTensor): The input sequence as (batch-size x in-seq-len x dim-model)-tensor.
            out_sequence (torch.FloatTensor): The output sequence as (batch-size x out-seq-len x dim-model)-tensor.
            padding_mask (torch.ByteTensor): A padding mask as (batch-size x in-seq-len x in-seq-len)-tensor or
                ``None`` if no mask is used.
            shifted_output_mask (torch.ByteTensor): The shifted-output mask as
                (batch-size x out-seq-len x in-seq-len)-tensor.

        Returns:
            FloatTensor: The computed outputs as (batch-size x out-seq-len x dim-model)-tensor.
        """
        # prepare mask for enc-dec attention
        if padding_mask is not None:
            if in_sequence.size(1) < out_sequence.size(1):
                padding_mask = padding_mask[:, :1, :].repeat(1, out_sequence.size(1), 1)
            elif in_sequence.size(1) > out_sequence.size(1):
                padding_mask = padding_mask[:, :out_sequence.size(1), :]
        
        # compute attention sub-layer 1
        out_sequence = self.norm(
                self.dropout(
                        self.attn_1(out_sequence, out_sequence, out_sequence, mask=shifted_output_mask)
                ) + out_sequence
        )
        
        # compute attention sub-layer 2
        out_sequence = self.norm(
                self.dropout(
                        self.attn_2(out_sequence, in_sequence, in_sequence, mask=padding_mask)
                ) + out_sequence
        )
        
        # compute feed-forward sub-layer
        out_sequence = self.norm(self.dropout(self.feed_forward(out_sequence)) + out_sequence)
        
        return out_sequence
    
    def reset_parameters(self) -> None:
        """Resets all trainable parameters of the module."""
        self.attn_1.reset_parameters()
        self.attn_2.reset_parameters()
        self.feed_forward.reset_parameters()
