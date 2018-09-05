# -*- coding: utf-8 -*-


import numbers


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


class EncDecBase(object):
    """A base class that implements common functionality of the encoder and decoder parts of the Transformer model."""

    def __init__(
            self,
            num_layers: int,
            num_heads: int,
            dim_model: int,
            dim_keys: int,
            dim_values: int,
            residual_dropout: numbers.Real,
            attention_dropout: numbers.Real,
            pad_index: int
    ):
        """Creates a new instance of ``EncDecBase``.
        
        Args:
            num_layers (int): The number of to use.
            num_heads (int): The number of attention heads to use.
            dim_model (int): The dimension to use for all layers. This is called d_model, in the paper.
            dim_keys (int): The size of the keys provided to the attention mechanism. This is called d_k, in the paper.
            dim_values (int): The size of the values provided to the attention mechanism. This is called d_v, in the
                paper.
            residual_dropout (numbers.Real): The dropout probability for residual connections (before they are added to
                the the sublayer output).
            attention_dropout (numbers.Real): The dropout probability for values provided by the attention mechanism.
            pad_index (int): The index that indicates a padding token in the input sequence.
        """
        super().__init__()
    
        # define attributes
        self._attention_dropout = None
        self._dim_keys = None
        self._dim_model = None
        self._dim_values = None
        self._num_heads = None
        self._num_layers = None
        self._pad_index = None
        self._residual_dropout = None
    
        # specify properties
        self.attention_dropout = attention_dropout
        self.dim_keys = dim_keys
        self.dim_model = dim_model
        self.dim_values = dim_values
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pad_index = pad_index
        self.residual_dropout = residual_dropout

    #  PROPERTIES  #####################################################################################################
    
    @property
    def attention_dropout(self) -> float:
        """float: The dropout probability for residual connections (before they are added to the the sublayer output).
        """
        return self._attention_dropout
    
    @attention_dropout.setter
    def attention_dropout(self, attention_dropout: numbers.Real):
        self._sanitize_probability("attention_dropout", attention_dropout)
        self._attention_dropout = float(attention_dropout)
    
    @property
    def dim_keys(self) -> int:
        """int: The size of the keys provided to the attention mechanism.
        
        This value is called d_k, in "Attention Is All You Need".
        """
        return self._dim_keys
    
    @dim_keys.setter
    def dim_keys(self, dim_keys: int) -> None:
        self._sanitize_pos_int("dim_keys", dim_keys)
        self._dim_keys = dim_keys

    @property
    def dim_model(self) -> int:
        """int: The dimension to use for all layers.

        This value is called d_model, in "Attention Is All You Need".
        """
        return self._dim_model

    @dim_model.setter
    def dim_model(self, dim_model: int) -> None:
        self._sanitize_pos_int("dim_model", dim_model)
        self._dim_model = dim_model

    @property
    def dim_values(self) -> int:
        """int: The size of the values provided to the attention mechanism.

        This value is called d_v, in "Attention Is All You Need".
        """
        return self._dim_values

    @dim_values.setter
    def dim_values(self, dim_values: int) -> None:
        self._sanitize_pos_int("dim_values", dim_values)
        self._dim_values = dim_values

    @property
    def num_heads(self) -> int:
        """int: The number of attention heads used by the implemented module."""
        return self._num_heads

    @num_heads.setter
    def num_heads(self, num_heads: int) -> None:
        self._sanitize_pos_int("num_heads", num_heads)
        self._num_heads = num_heads
    
    @property
    def num_layers(self) -> int:
        """int: The number of layers used by the implemented module."""
        return self._num_layers
    
    @num_layers.setter
    def num_layers(self, num_layers: int) -> None:
        self._sanitize_pos_int("num_layers", num_layers)
        self._num_layers = num_layers
    
    @property
    def pad_index(self) -> int:
        """int: The index that indicates a padding token in the input sequence."""
        return self._pad_index
    
    @pad_index.setter
    def pad_index(self, pad_index: int) -> None:
        if not isinstance(pad_index, int):
            raise TypeError("<pad_index> has to be an integer!")
        if pad_index < 0:
            raise ValueError("<pad_index> has to be non-negative!")
        self._pad_index = pad_index

    @property
    def residual_dropout(self) -> float:
        """float: The dropout probability for values provided by the attention mechanism."""
        return self._residual_dropout

    @residual_dropout.setter
    def residual_dropout(self, residual_dropout: numbers.Real):
        self._sanitize_probability("residual_dropout", residual_dropout)
        self._residual_dropout = float(residual_dropout)
    
    #  METHODS  ########################################################################################################
    
    @staticmethod
    def _sanitize_pos_int(arg_name: str, arg_value) -> None:
        """Ensures that the provided arg is a positive integer.
        
        Args:
            arg_name (str): The name of the arg being sanitized.
            arg_value: The value being sanitized.
        
        Raises:
            TypeError: If ``arg_value`` is not an ``int``.
            ValueError: If ``arg_value`` is not a positive number.
        """
        if not isinstance(arg_value, int):
            raise TypeError("<{}> has to be an integer!".format(arg_name))
        if arg_value < 1:
            raise ValueError("<{}> has to be > 0!".format(arg_name))
    
    @staticmethod
    def _sanitize_probability(arg_name: str, arg_value):
        """Ensures that the provided arg is a probability.
        
        Args:
            arg_name (str): The name of the arg being sanitized.
            arg_value: The value being sanitized.
        
        Raises:
            TypeError: If ``arg_value`` is not a ``numbers.Real``.
            ValueError: If ``arg_value`` is not in [0, 1].
        """
        if not isinstance(arg_value, numbers.Real):
            raise TypeError("<{}> has to be a real number!".format(arg_name))
        if arg_value < 0 or float(arg_value) > 1:
            raise ValueError("<{}> has to be in [0, 1]!".format(arg_name))
