# -*- coding: utf-8 -*-


import torch
import torchtestcase as ttc

from unittest import mock

from torch import nn

from transformer import transformer
from transformer import transformer_tools as tt


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
__date__ = "Aug 30, 2018"
__maintainer__ = "Patrick Hohenecker"
__email__ = "mail@paho.at"
__status__ = "Development"


class TransformerToolsTest(ttc.TorchTestCase):
    
    TOLERANCE = 1e-6
    
    def test_eval_probability(self):
        pad_index = 0
        
        # create the test data
        input_seq = torch.LongTensor(  # the actual values are irrelevant, since the model's forward method is mocked
                [
                        [1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 1, 0],
                        [1, 1, 0, 0, 0, 0]
                ]
        )
        target_seq = torch.LongTensor(
                [
                        [1, 3, 0],
                        [4, 2, 1],
                        [2, 0, 0]
                ]
        )
        mock_probs = torch.FloatTensor(
                [
                        [
                                [0.2, 0.2, 0.2, 0.2, 0.2],
                                [0.05, 0.05, 0.05, 0.8, 0.05],
                                [0.4, 0.3, 0.2, 0.05, 0.05]
                        ],
                        [
                                [0.2, 0.2, 0.2, 0.2, 0.2],
                                [0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.1, 0.6, 0.05, 0.05, 0.2]
                        ],
                        [
                            [0.2, 0.2, 0.2, 0.2, 0.2],
                            [0.0, 0.1, 0.2, 0.3, 0.4],
                            [0.4, 0.3, 0.2, 0.1, 0.0]
                        ]
                ]
        )
        target_probs = torch.FloatTensor([0.16, 0.12, 0.2])
        
        
        # prepare model
        model = transformer.Transformer(
                nn.Embedding(5, 5),
                pad_index,
                10,
                max_seq_len=10
        )
        model.forward = mock.MagicMock(return_value=mock_probs)

        # evaluate the probabilities of the data as predicted by the model
        probs = tt.eval_probability(model, input_seq, target_seq, pad_index=pad_index)
        
        # CHECK: the probabilities are as expected
        self.assertLessEqual(
                (target_probs - probs).abs(),
                self.TOLERANCE
        )
