# -*- coding: utf-8 -*-


import typing

import numpy as np
import torch

from torch import nn
from torch.nn import init


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


class MultiHeadAttention(nn.Module):
    """A multi-head scaled dot-product attention mechanism as it is used in *Attention Is All You Need*."""
    
    def __init__(self, num_heads: int, dim_model: int, dim_keys: int, dim_values: int, dropout_rate: float):
        """Creates a new instance of ``MultiHeadAttention``.
        
        Notice:
            This constructor does not sanitize any parameters, which means that this has to be taken care of beforehand.

        Args:
            num_heads (int): The number of attention heads to use.
            dim_model (int): The dimension used for all layers in the model that the ``MultiHeadAttention`` belongs to.
            dim_keys (int): The target size to project keys to.
            dim_values (int): The target size to project values to.
            dropout_rate (float): The dropout probability to use.
        """
        super().__init__()
        
        # store all of the provided args
        self.dim_keys = dim_keys
        self.dim_model = dim_model
        self.dim_values = dim_values
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        
        # create projections for inputs
        self.query_projection = nn.Parameter(torch.empty(self.num_heads, self.dim_model, self.dim_keys))
        self.key_projection = nn.Parameter(torch.empty(self.num_heads, self.dim_model, self.dim_keys))
        self.value_projection = nn.Parameter(torch.empty(self.num_heads, self.dim_model, self.dim_values))
        
        # create output projection
        self.output_projection = nn.Parameter(torch.empty(self.num_heads * self.dim_values, self.dim_model))
        
        # create softmax and dropout layers
        self.dropout = nn.Dropout(self.dropout_rate)
        self.softmax = nn.Softmax(dim=3)
        
        # initialize all parameters
        self.reset_parameters()
    
    #  METHODS  ########################################################################################################
    
    def _apply_attention(
            self,
            queries: torch.FloatTensor,
            keys: torch.FloatTensor,
            values: torch.FloatTensor,
            mask: typing.Optional[torch.ByteTensor]
    ) -> torch.Tensor:
        """The actual attention mechanism.
        
        Args:
            queries (torch.FloatTensor): The queries as (batch_size x num_heads x Q x dim_keys)-tensor.
            keys (torch.FloatTensor): The keys as (batch_size x num_heads x KV x dim_keys)-tensor.
            values (torch.FloatTensor): The values as (batch_size x num_heads x KV x dim_values)-tensor.
            mask (torch.ByteTensor): An optional binary mask that indicates which key-value pairs to consider for each
                of the queries. If provided, then this has to be a (batch_size x Q x KV)-tensor.
        
        Returns:
            torch.FloatTensor: The computed "attended" values as (batch_size x num_heads x Q x dim_values)-tensor. If
                the ``mask`` specifies that none of the key-value pairs shall be used for any of the queries, then the
                according attended value is set to ``0``.
        """
        # compute inputs to the softmax
        attn = queries.matmul(keys.transpose(2, 3)) / np.sqrt(self.dim_keys)  # compute (Q * K^T) / sqrt(d_k)
        # -> (batch_size x num_heads x Q x KV)
        
        # apply the mask (if provided)
        if mask is not None:
            
            # check whether the mask excludes all of the entries
            if mask.sum().item() == 0:
                return torch.zeros(queries.size())
            
            # expand mask to cover all heads
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

            # determine which token masks are all-0
            non_zero_parts = (mask.sum(dim=-1) != 0).unsqueeze(-1).expand(*mask.size())
            
            # remove the all-0 parts from the original mask
            mask = 1 - (1 - mask) * non_zero_parts
            
            # apply mask
            attn.masked_fill_(1 - mask, -np.inf)

            # compute attention scores
            attn = self.softmax(attn)
            
            # apply all-0 parts of the masks
            attn = attn * non_zero_parts.float()
        else:
            # compute attention scores
            attn = self.softmax(attn)

        # apply dropout
        attn = self.dropout(attn)

        # compute attended value
        return attn.matmul(values)  # -> (batch_size x num_heads x Q x dim_values)
    
    def _project_inputs(
            self,
            queries: torch.FloatTensor,
            keys: torch.FloatTensor,
            values: torch.FloatTensor
    ) -> typing.Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor
    ]:
        """Projects all inputs provided to the attention mechanism to the needed sizes.
        
        This means that queries and keys are projected from ``dim_model`` to ``dim_keys``, and values from ``dim_model``
        to ``dim_values``.
        
        Args:
            queries (torch.FloatTensor): The queries as (batch_size x Q x dim_model)-tensor.
            keys (torch.FloatTensor): The keys as (batch_size x KV x dim_model)-tensor.
            values (torch.FloatTensor): The values as (batch_size x KV x dim_model)-tensor.
        
        Returns:
            tuple: A triple of ``FloatTensor``s, consisting of the projected queries, keys, and values.
        """
        # for each of the attention heads, project inputs to the needed dimensions
        queries = queries.unsqueeze(1).matmul(self.query_projection)  # -> (batch_size x num_heads x Q  x dim_keys)
        keys = keys.unsqueeze(1).matmul(self.key_projection)          # -> (batch_size x num_heads x KV x dim_keys)
        values = values.unsqueeze(1).matmul(self.value_projection)    # -> (batch_size x num_heads x KV x dim_values)
        
        return queries, keys, values
    
    def _project_output(self, attn_values: torch.FloatTensor) -> torch.FloatTensor:
        """Projects the "attended" values of all heads to the required output size.
        
        Args:
            attn_values (torch.FloatTensor): The attended values as (batch_size x num_heads x Q x dim_values)-tensor.
        
        Returns:
            torch.FloatTensor: The computed output as (batch_size x Q x dim_model)-tensor.
        """
        # concatenate the values retrieved from all heads
        batch_size = attn_values.size(0)
        num_queries = attn_values.size(2)
        attn_values = attn_values.transpose(1, 2).reshape(batch_size, num_queries, -1)
        # -> (batch_size x Q x (num_heads * dim_values))

        return attn_values.matmul(self.output_projection)  # -> (batch-size x Q x dim_model)
    
    def forward(
            self,
            queries: torch.FloatTensor,
            keys: torch.FloatTensor,
            values: torch.FloatTensor,
            mask: torch.ByteTensor=None
    ) -> torch.Tensor:
        """Runs the attention mechanism.
        
        Args:
            queries (torch.FloatTensor): The queries as (batch_size x Q x dim_model)-tensor.
            keys (torch.FloatTensor): The keys as (batch_size x KV x dim_model)-tensor.
            values (torch.FloatTensor): The values as (batch_size x KV x dim_model)-tensor.
            mask (torch.ByteTensor, optional): An optional binary mask that indicates which key-value pairs to consider
                for each of the queries. If provided, then this has to be a (batch_size x Q x KV)-tensor.
        
        Returns:
            torch.FloatTensor: The values computed by the attention mechanism as (batch_size x Q x dim_model)-tensor.
        """
        assert isinstance(queries, torch.FloatTensor) or isinstance(queries, torch.cuda.FloatTensor)
        assert isinstance(keys, torch.FloatTensor) or isinstance(keys, torch.cuda.FloatTensor)
        assert isinstance(values, torch.FloatTensor) or isinstance(values, torch.cuda.FloatTensor)
        assert queries.dim() == 3
        assert keys.dim() == 3
        assert values.dim() == 3
        assert queries.size(0) == keys.size(0)
        assert queries.size(0) == values.size(0)
        assert queries.size(2) == keys.size(2)
        assert queries.size(2) == values.size(2)
        assert keys.size(1) == values.size(1)
        if mask is not None:
            assert isinstance(mask, torch.ByteTensor) or isinstance(mask, torch.cuda.ByteTensor)
            assert mask.dim() == 3
            assert queries.size(0) == mask.size(0)
            assert queries.size(1) == mask.size(1)
            assert keys.size(1) == mask.size(2)
        
        # for each of the attention heads, project inputs to the needed dimensions
        queries, keys, values = self._project_inputs(queries, keys, values)
        
        # compute attention value
        attn_values = self._apply_attention(queries, keys, values, mask)
        
        # project retrieved values to needed dimensions
        return self._project_output(attn_values)
    
    def reset_parameters(self):
        """Resets all trainable parameters of the module."""
        init.xavier_normal_(self.query_projection)
        init.xavier_normal_(self.key_projection)
        init.xavier_normal_(self.value_projection)
        init.xavier_normal_(self.output_projection)
