# -*- coding: utf-8 -*-


import torch
import torchtestcase as ttc

from transformer import decoder
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
__date__ = "Aug 29, 2018"
__maintainer__ = "Patrick Hohenecker"
__email__ = "mail@paho.at"
__status__ = "Development"


class DecoderTest(ttc.TorchTestCase):
    
    TOLERANCE = 1e-4
    """float: The tolerance used for tensor equality assertions."""
    
    def setUp(self):
        self.num_layers = 2
        self.num_heads = 4
        self.dim_model = 5
        self.dim_keys = 3
        self.dim_values = 3
        self.residual_dropout = 0.1
        self.attention_dropout = 0.2
        self.pad_index = 0
    
    def test_forward(self):
        dec = decoder.Decoder(
                self.num_layers,
                self.num_heads,
                self.dim_model,
                self.dim_keys,
                self.dim_values,
                0,  # residual dropout
                0,  # attention dropout
                self.pad_index
        )
        
        # create test data
        input_seq = torch.FloatTensor(
                [
                        [
                                [1, 2, 3, 4, 5],
                                [11, 22, 33, 44, 55],
                                [111, 222, 333, 444, 555]
                        ],
                        [
                                [6, 7, 8, 9, 0],
                                [66, 77, 88, 99, 00],
                                [666, 777, 888, 999, 000]
                        ]
                ]
        )
        padding_mask = torch.ones(2, 3, 3).byte()
        output_seq = torch.FloatTensor(
                [
                        [
                                [0.1, 0.2, 0.3, 0.4, 0.5],
                                [0.11, 0.22, 0.33, 0.44, 0.55],
                                [0.111, 0.222, 0.333, 0.444, 0.555],
                                [0.1111, 0.2222, 0.3333, 0.4444, 0.5555]
                        ],
                        [
                                [0.6, 0.7, 0.8, 0.9, 0.0],
                                [0.66, 0.77, 0.88, 0.99, 0.00],
                                [0.666, 0.777, 0.888, 0.999, 0.000],
                                [0.6666, 0.7777, 0.8888, 0.9999, 0.0000]
                        ]
                ]
        )
        output_seq_2 = output_seq.clone()
        output_seq_2[0, -1] = torch.FloatTensor([0.6, 0.7, 0.8, 0.9, 1.0])
        output_seq_2[1, -1] = torch.FloatTensor([0.66, 0.77, 0.88, 0.99, 11.0])
        shifted_output_mask = util.create_shifted_output_mask(output_seq)
        
        # NOTICE:
        # output_seq and output_seq_1 differ at the last time step only

        # compute target
        target = util.shift_output_sequence(output_seq)
        for layer in dec._layers:
            target = layer(input_seq, target, padding_mask, shifted_output_mask)

        # run the decoder on both output sequences
        dec_seq = dec(input_seq, output_seq, padding_mask)
        dec_seq_2 = dec(input_seq, output_seq_2, padding_mask)

        # CHECK: the provided output has the same shape as the input
        self.assertEqual(output_seq.size(), dec_seq.size())

        # CHECK: the encoder computes the expected targets
        self.eps = self.TOLERANCE
        self.assertEqual(target, dec_seq)
        self.assertEqual(target, dec_seq_2)
        # -> both of the computed values have to be equal, as the provided target output sequences differ at the last
        #    time step only, which is not considered because of the performed shift

    def test_init(self):
        dec = decoder.Decoder(
                self.num_layers,
                self.num_heads,
                self.dim_model,
                self.dim_keys,
                self.dim_values,
                self.residual_dropout,
                self.attention_dropout,
                self.pad_index
        )
    
        # CHECK: the correct number of layers was created
        self.assertEqual(self.num_layers, len(dec._layers))
    
        for layer in dec._layers:  # iterate over all layers in the encoder

            # CHECK: the first attention mechanism was configured correctly
            self.assertEqual(self.dim_keys, layer.attn_1.dim_keys)
            self.assertEqual(self.dim_model, layer.attn_1.dim_model)
            self.assertEqual(self.dim_values, layer.attn_1.dim_values)
            self.assertEqual(self.attention_dropout, layer.attn_1.dropout_rate)
            self.assertEqual(self.num_heads, layer.attn_1.num_heads)

            # CHECK: the second attention mechanism was configured correctly
            self.assertEqual(self.dim_keys, layer.attn_2.dim_keys)
            self.assertEqual(self.dim_model, layer.attn_2.dim_model)
            self.assertEqual(self.dim_values, layer.attn_2.dim_values)
            self.assertEqual(self.attention_dropout, layer.attn_2.dropout_rate)
            self.assertEqual(self.num_heads, layer.attn_2.num_heads)

            # CHECK: the feed-forward layer was configured correctly
            self.assertEqual(self.dim_model, layer.feed_forward.dim_model)

            # CHECK: the dropout mechanism uses the correct dropout rate
            self.assertEqual(self.residual_dropout, layer.dropout.p)
