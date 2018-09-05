# -*- coding: utf-8 -*-


import torch

from torch import nn
from torch.nn import functional


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
__date__ = "Aug 22, 2018"
__maintainer__ = "Patrick Hohenecker"
__email__ = "mail@paho.at"
__status__ = "Development"


class FeedForwardLayer(nn.Module):
    """A sublayer that computes a 1-hidden-layer multi-layer perceptron for each token in a sequences."""
    
    def __init__(self, dim_model: int):
        """Creates a new instance of ``FeedForwardLayer``.
        
        Args:
             dim_model (int): The dimension of all tokens in the input sequence. This is called d_model, in the paper.
        """
        super().__init__()
        
        # sanitize args
        if not isinstance(dim_model, int):
            raise TypeError("<dim_model> has to be an integer!")
        if dim_model < 1:
            raise ValueError("<dim_model> has to be a positive number!")
        
        # store arg
        self._dim_model = dim_model
        
        # create layers
        self._layer_1 = nn.Conv1d(self._dim_model, self._dim_model, 1)
        self._layer_2 = nn.Conv1d(self._dim_model, self._dim_model, 1)
    
    #  PROPERTIES  #####################################################################################################
    
    @property
    def dim_model(self) -> int:
        """int: The dimension of all tokens in the input sequence.
        
        This is called d_model, in the paper.
        """
        return self._dim_model
    
    @property
    def layer_1(self) -> nn.Conv1d:
        """nn.Conv1d: The first linear layer (before the ReLU non-linearity is applied)."""
        return self._layer_1
    
    @property
    def layer_2(self) -> nn.Conv1d:
        """nn.Conv1d: The second linear layer."""
        return self._layer_2
    
    #  METHODS  ########################################################################################################
    
    def forward(self, sequence: torch.FloatTensor) -> torch.FloatTensor:
        """Runs the feed-forward layer.
        
        Args:
            sequence (torch.FloatTensor): The input sequence given as (batch_size x seq_len x dim_model)-tensor.
        
        Returns:
            torch.FloatTensor: The computed values as (batch_size x seq_len x dim_model)-tensor.
        """
        assert sequence.dim() == 3
        assert sequence.size(2) == self._dim_model

        sequence = functional.relu(self._layer_1(sequence.transpose(1, 2)))
        sequence = self._layer_2(sequence).transpose(1, 2)
        
        return sequence
    
    def reset_parameters(self):
        """Resets all trainable parameters of the module."""
        self._layer_1.reset_parameters()
        self._layer_2.reset_parameters()
