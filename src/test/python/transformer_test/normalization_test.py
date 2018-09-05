# -*- coding: utf-8 -*-


import torch
import torchtestcase as ttc

from transformer import normalization


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


class NormalizationTest(ttc.TorchTestCase):

    def test_forward(self):
        # create test data
        data = torch.FloatTensor(
                [
                        [
                                [1, 2, 3],
                                [11, 22, 33]
                        ],
                        [
                                [111, 222, 333],
                                [1111, 2222, 3333]
                        ]
                ]
        )
        
        # create layer used for testing
        norm = normalization.Normalization()
        
        # run layer to normalize the data
        norm_data = norm(data)
        
        for sample_idx in range(data.size(0)):  # iterate over all samples in the batch
            for token_idx in range(data.size(1)):  # iterate over all tokens in the sequences
                
                # normalize current token
                norm_token = data[sample_idx, token_idx]
                norm_token = norm_token - torch.mean(norm_token)
                norm_token = norm_token / (torch.std(norm_token) + norm.eps)
                
                # CHECK: the data has been normed correctly
                self.assertLessEqual(
                        (norm_token - norm_data[sample_idx, token_idx]).abs(),
                        torch.ones(norm_token.size()) * norm.eps
                )
    
        # run layer on 0-data
        norm_data = norm(torch.zeros(1, 2, 3))
        
        # CHECK: the data is approximately zero
        self.assertLessEqual(norm_data.max().item(), norm.eps)
