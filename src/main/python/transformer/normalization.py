# -*- coding: utf-8 -*-


import numbers

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
__date__ = "Aug 21, 2018"
__maintainer__ = "Patrick Hohenecker"
__email__ = "mail@paho.at"
__status__ = "Development"


class Normalization(nn.Module):
    """A normalization layer."""
    
    def __init__(self, eps: numbers.Real=1e-15):
        """Creates a new instance of ``Normalization``.
        
        Args:
            eps (numbers.Real, optional): A tiny number to be added to the standard deviation before re-scaling the
                centered values. This prevents divide-by-0 errors. By default, this is set to ``1e-15``.
        """
        super().__init__()
    
        self._eps = None
        self.eps = float(eps)
    
    #  PROPERTIES  #####################################################################################################
    
    @property
    def eps(self) -> float:
        """float: A tiny number that is added to the standard deviation before re-scaling the centered values.
        
        This prevents divide-by-0 errors. By default, this is set to ``1e-15``.
        """
        return self._eps
    
    @eps.setter
    def eps(self, eps: numbers.Real) -> None:
        if not isinstance(eps, numbers.Real):
            raise TypeError("<eps> has to be a real number!")
        self._eps = float(eps)
    
    #  METHODS  ########################################################################################################

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Runs the normalization layer.
        
        Args:
            x (torch.FloatTensor): A tensor to be normalized. To that end, ``x`` is interpreted as a batch of values
                where normalization is applied over the last of its dimensions.
        
        Returns:
            torch.FloatTensor: The normalized tensor.
        """
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        
        return (x - mean) / (std + self._eps)
