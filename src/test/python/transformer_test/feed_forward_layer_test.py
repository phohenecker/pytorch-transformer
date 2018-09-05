# -*- coding: utf-8 -*-


import torch
import torchtestcase as ttc

from torch.nn import functional

from transformer import feed_forward_layer as ffl


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
__date__ = "Aug 23, 2018"
__maintainer__ = "Patrick Hohenecker"
__email__ = "mail@paho.at"
__status__ = "Development"


class FeedForwardLayerTest(ttc.TorchTestCase):
    
    TOLERANCE = 1e-5
    
    def test_forward(self):
        dim_model = 5
        
        # create model for testing
        ff_layer = ffl.FeedForwardLayer(dim_model)
        
        # create test data
        input_seq = torch.FloatTensor(
                [
                        [
                                [1, 2, 3, 4, 5],
                                [6, 7, 8, 9, 0]
                        ],
                        [
                                [11, 22, 33, 44, 55],
                                [66, 77, 88, 99, 00]
                        ]
                ]
        )
        
        # fetch parameters of the feed-forward layer
        weight_1 = ff_layer.layer_1.weight.data
        bias_1 = ff_layer.layer_1.bias.data.data
        weight_2 = ff_layer.layer_2.weight.data
        bias_2 = ff_layer.layer_2.bias.data.data
        
        # CHECK: the parameters are as expected
        self.assertEqual((dim_model, dim_model, 1), weight_1.size())
        self.assertEqual((dim_model,), bias_1.size())
        self.assertEqual((dim_model, dim_model, 1), weight_2.size())
        self.assertEqual((dim_model,), bias_2.size())
        
        # turn all parameters into matrices
        weight_1 = weight_1.squeeze()
        bias_1 = bias_1.unsqueeze(1)
        weight_2 = weight_2.squeeze()
        bias_2 = bias_2.unsqueeze(1)
        
        # run the feed-forward layer
        output = ff_layer(input_seq).data
        
        # CHECK: the output has the same shape as the input
        self.assertEqual(input_seq.size(), output.size())
        
        for sample_idx in range(input_seq.size(0)):  # iterate over all samples in the batch
            for token_idx in range(input_seq.size(1)):  # iterate over all tokens in the input sequences
                
                # compute target value
                target = input_seq[sample_idx, token_idx].unsqueeze(1)
                target = weight_1.matmul(target) + bias_1
                target = functional.relu(target)
                target = weight_2.matmul(target) + bias_2
                target = target.squeeze()
                
                # CHECK: the corresponding token has been processed correctly
                self.assertLessEqual(
                        (target - output[sample_idx, token_idx]).abs(),
                        torch.ones(target.size()) * self.TOLERANCE
                )
