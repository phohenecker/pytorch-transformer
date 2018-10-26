# -*- coding: utf-8 -*-


import numbers

import torch

from torch import nn
from torch.nn import functional

from transformer import decoder as dec
from transformer import encoder as enc
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


class Transformer(nn.Module):
    """The Transformer model that was introduced in *Attention Is All You Need*."""
    
    def __init__(
            self,
            word_emb: nn.Embedding,
            pad_index: int,
            output_size: int,
            positional_emb: nn.Embedding=None,
            max_seq_len: int=None,
            num_layers: int=6,
            num_heads: int=8,
            dim_model: int=512,
            dim_keys: int=64,
            dim_values: int=64,
            residual_dropout: numbers.Real=0.1,
            attention_dropout: numbers.Real=0.1
    ):
        """Creates a new instance of ``Transformer``.
        
        Args:
            word_emb (nn.Embedding): The word embeddings to use.
            pad_index (int): The index that indicates that a token in an input sequence is just padding.
            output_size (int): The size, i.e., the number of dimensions, of the output to provide.
            positional_emb (nn.Embedding, optional): The positional embeddings to use.
            max_seq_len (int, optional): The maximum length of any input or output sequences. This is used to generate
                positional embeddings, if ``positional_emb`` is not provided.
            num_layers (int): The number of to use.
            num_heads (int): The number of attention heads to use.
            dim_model (int): The dimension to use for all layers. This is called d_model, in the paper.
            dim_keys (int): The size of the keys provided to the attention mechanism. This is called d_k, in the paper.
            dim_values (int): The size of the values provided to the attention mechanism. This is called d_v, in the
                paper.
            residual_dropout (numbers.Real): The dropout probability for residual connections (before they are added to
                the the sublayer output).
            attention_dropout (numbers.Real): The dropout probability for values provided by the attention mechanism.
        """
        super().__init__()
        
        # sanitize args
        if not isinstance(word_emb, nn.Embedding):
            raise TypeError("<word_emb> has to be an instance of torch.nn.Embedding!")
        if not isinstance(output_size, int):
            raise TypeError("The <output_size> has to be an integer!")
        if output_size < 1:
            raise ValueError("The <output_size> has to be a positive number!")
        if positional_emb is not None:
            if not isinstance(positional_emb, nn.Embedding):
                raise TypeError("<positional_emb> has to be an instance of torch.nn.Embedding!")
            if word_emb.embedding_dim != positional_emb.embedding_dim:
                raise ValueError("<word_emb> and <positional_emb> have to use the same embedding size!")
        if max_seq_len is not None:
            if not isinstance(max_seq_len, int):
                raise TypeError("The <max_seq_len> has to be an integer!")
            if max_seq_len < 1:
                raise ValueError("<max_seq_len> has to be a positive number!")
            elif positional_emb is not None and max_seq_len > positional_emb.num_embeddings:
                raise ValueError("<max_seq_len> cannot be greater than the number of embeddings in <positional_emb>!")
        elif positional_emb is None:
                raise ValueError("At least one of the args <positional_emb> and <max_seq_len> has to be provided!")
        
        # store output_size and pad_index
        self._output_size = output_size
        self._pad_index = pad_index
        self._word_emb = word_emb
        
        # create encoder and decoder
        # (these are created first, because they sanitize all of the provided args)
        self._encoder = enc.Encoder(
                num_layers,
                num_heads,
                dim_model,
                dim_keys,
                dim_values,
                residual_dropout,
                attention_dropout,
                pad_index
        )
        self._decoder = dec.Decoder(
                num_layers,
                num_heads,
                dim_model,
                dim_keys,
                dim_values,
                residual_dropout,
                attention_dropout,
                pad_index
        )

        # store embeddings
        if positional_emb is None:
            self._positional_emb = util.create_positional_emb(max_seq_len, word_emb.embedding_dim, dim_model)
        else:
            self._positional_emb = positional_emb

        # figure out the maximum sequence length
        self._max_seq_len = self._positional_emb.num_embeddings
        
        # create linear projections for input (word embeddings) and output
        self._input_projection = nn.Linear(self._word_emb.embedding_dim, dim_model)
        self._output_projection = nn.Linear(dim_model, self._output_size)
    
    #  PROPERTIES  #####################################################################################################
    
    @property
    def decoder(self) -> dec.Decoder:
        """:class:`dec.Decoder`: The decoder part of the Transformer."""
        return self._decoder
    
    @property
    def embedding_dim(self) -> int:
        """int: The used embedding size."""
        return self._word_emb.embedding_dim
    
    @property
    def encoder(self) -> enc.Encoder:
        """:class:`enc.Encoder`: The encoder part of the Transformer."""
        return self._encoder
    
    @property
    def input_projection(self) -> nn.Linear:
        """nn.Linear: The linear projection between input and encoder."""
        return self._input_projection
    
    @property
    def max_seq_len(self) -> int:
        """int: The maximum length that any input sequence may have."""
        return self._max_seq_len

    @property
    def output_projection(self) -> nn.Linear:
        """nn.Linear: The linear projection between decoder and output."""
        return self._output_projection
    
    @property
    def output_size(self) -> int:
        """int: The size of the output provided by the ``Transformer``."""
        return self._output_size
    
    @property
    def pad_index(self) -> int:
        """int: The index that indicates that a token in an input sequence is just padding."""
        return self._pad_index
    
    @property
    def positional_emb(self):
        """nn.Embedding: The used positional embeddings."""
        return self._positional_emb
    
    @property
    def word_emb(self) -> nn.Embedding:
        """nn.Embedding: The used word embeddings."""
        return self._word_emb
    
    #  METHODS  ########################################################################################################
    
    def forward(self, input_seq: torch.LongTensor, target: torch.LongTensor) -> torch.FloatTensor:
        """Runs the Transformer.
        
        The Transformer expects both an input as well as a target sequence to be provided, and yields a probability
        distribution over all possible output tokens for each position in the target sequence.
        
        Args:
            input_seq (torch.LongTensor): The input sequence as (batch-size x input-seq-len)-tensor.
            target (torch.LongTensor): The target sequence as (batch-size x target-seq-len)-tensor.
        
        Returns:
            torch.FloatTensor: The computed probabilities for each position in ``target`` as a
                (batch-size x target-seq-len x output-size)-tensor.
        """
        # sanitize args
        if not isinstance(input_seq, torch.LongTensor) and not isinstance(input_seq, torch.cuda.LongTensor):
            raise TypeError("<input_seq> has to be a LongTensor!")
        if input_seq.dim() != 2:
            raise ValueError("<input_seq> has to have 2 dimensions!")
        if not isinstance(target, torch.LongTensor) and not isinstance(target, torch.cuda.LongTensor):
            raise TypeError("<target> has to be a LongTensor!")
        if target.dim() != 2:
            raise ValueError("<target> has to have 2 dimensions!")
        
        # create a tensor of indices, which is used to retrieve the according positional embeddings below
        index_seq = input_seq.new(range(input_seq.size(1))).unsqueeze(0).expand(input_seq.size(0), -1)
        
        # create padding mask for input
        padding_mask = util.create_padding_mask(input_seq, self._pad_index)
        
        # embed the provided input
        input_seq = self._word_emb(input_seq) + self._positional_emb(index_seq)
        
        # project input to the needed size
        input_seq = self._input_projection(input_seq)
        
        # run the encoder
        input_seq = self._encoder(input_seq, padding_mask=padding_mask)
        
        # create a tensor of indices, which is used to retrieve the positional embeddings for the targets below
        index_seq = target.new(range(target.size(1))).unsqueeze(0).expand(target.size(0), -1)
        
        # embed the provided targets
        target = self._word_emb(target) + self._positional_emb(index_seq)

        # project target to the needed size
        target = self._input_projection(target)
        
        # run the decoder
        output = self._decoder(input_seq, target, padding_mask=padding_mask)
        
        # project output to the needed size
        output = self._output_projection(output)
        
        # compute softmax
        return functional.softmax(output, dim=2)
    
    def reset_parameters(self) -> None:
        """Resets all trainable parameters of the module."""
        self._encoder.reset_parameters()
        self._decoder.reset_parameters()
        self._input_projection.reset_parameters()
        self._output_projection.reset_parameters()
