# -*- coding: utf-8 -*-


import torch
import torchtestcase as ttc

from transformer import encoder


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


class EncoderTest(ttc.TorchTestCase):
    
    TOLERANCE = 1e-5
    
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
        enc = encoder.Encoder(
                self.num_layers,
                self.num_heads,
                self.dim_model,
                self.dim_keys,
                self.dim_values,
                0,  # residual dropout
                0,   # attention dropout,
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
        
        # compute target
        target = input_seq
        for layer in enc._layers:
            target = layer(target, padding_mask)
        target = target.detach()
        
        # run the encoder
        enc_seq = enc(input_seq).detach()
        
        # CHECK: the provided output has the same shape as the input
        self.assertEqual(input_seq.size(), enc_seq.size())
        
        # CHECK: the encoder computes the expected target
        self.assertLessEqual(
                (target - enc_seq).abs(),
                torch.ones(target.size()) * self.TOLERANCE
        )
    
    def test_init(self):
        enc = encoder.Encoder(
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
        self.assertEqual(self.num_layers, len(enc._layers))
        
        for layer in enc._layers:  # iterate over all layers in the encoder
            
            # CHECK: the attention mechanism was configured correctly
            self.assertEqual(self.dim_keys, layer.attn.dim_keys)
            self.assertEqual(self.dim_model, layer.attn.dim_model)
            self.assertEqual(self.dim_values, layer.attn.dim_values)
            self.assertEqual(self.attention_dropout, layer.attn.dropout_rate)
            self.assertEqual(self.num_heads, layer.attn.num_heads)

            # CHECK: the feed-forward layer was configured correctly
            self.assertEqual(self.dim_model, layer.feed_forward.dim_model)
            
            # CHECK: the dropout mechanism uses the correct dropout rate
            self.assertEqual(self.residual_dropout, layer.dropout.p)
