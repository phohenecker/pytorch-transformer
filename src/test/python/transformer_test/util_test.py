# -*- coding: utf-8 -*-


import numpy as np
import torch
import torchtestcase as ttc

from torch import nn

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


class UtilTest(ttc.TorchTestCase):
    
    TOLERANCE = 1e-5
    
    def test_create_padding_mask(self):
        # create test data
        seq = torch.LongTensor(
                [
                        [4, 2, 3, 5, 1, 0],
                        [1, 4, 0, 0, 0, 0],
                        [6, 3, 2, 4, 5, 1],
                        [5, 3, 2, 3, 0, 0],
                        [0, 0, 0, 0, 0, 0]
                ]
        )
        target_mask = torch.ByteTensor(
                [
                        [
                                [1, 1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1, 0]
                        ],
                        [
                                [1, 1, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0]
                        ],
                        [
                                [1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1]
                        ],
                        [
                                [1, 1, 1, 1, 0, 0],
                                [1, 1, 1, 1, 0, 0],
                                [1, 1, 1, 1, 0, 0],
                                [1, 1, 1, 1, 0, 0],
                                [1, 1, 1, 1, 0, 0],
                                [1, 1, 1, 1, 0, 0]
                        ],
                        [
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0]
                        ]
                ]
        )
        
        # create the padding mask
        mask = util.create_padding_mask(seq, 0)
        
        # CHECK: the retrieved mask is a ByteTensor
        self.assertIsInstance(mask, torch.ByteTensor)
        
        # CHECK: the mask has the correct shape
        self.assertEqual((seq.size(0), seq.size(1), seq.size(1)), mask.size())
        
        # CHECK: the mask contains the correct values
        self.assertEqual(target_mask, mask)

    def test_create_positional_emb(self):
        dim_model = 512
        embedding_size = 300
        max_seq_len = 100
    
        # create positional embeddings
        pos_emb = util.create_positional_emb(max_seq_len, embedding_size, dim_model)
    
        # CHECK: the retrieved instance is an embedding of correct size
        self.assertIsInstance(pos_emb, nn.Embedding)
        self.assertEqual(max_seq_len, pos_emb.num_embeddings)
        self.assertEqual(embedding_size, pos_emb.embedding_dim)
    
        for pos in range(max_seq_len):  # iterate over the embeddings for all time steps
        
            # fetch embedding vector for the current index
            emb_vec = pos_emb(torch.ones(1, dtype=torch.long) * pos).squeeze()
        
            for i in range(embedding_size):  # iterate over all values of the current time step
            
                # fetch value for c
                emb_val = emb_vec[i].item()
            
                # CHECK: the considered value is the one expected
                if i % 2 == 0:
                    self.assertLessEqual(
                            np.abs(np.sin(pos / (10000 ** (i / dim_model))) - emb_val),
                            self.TOLERANCE
                    )
                else:
                    self.assertLessEqual(
                            np.abs(np.cos(pos / (10000 ** ((i - 1) / dim_model))) - emb_val),
                            self.TOLERANCE
                    )
    
    def test_create_shifted_output_mask(self):
        # create test data
        seq = torch.ones(2, 4).long()
        target_mask = torch.ByteTensor(
                [
                        [
                                [1, 0, 0, 0],
                                [1, 1, 0, 0],
                                [1, 1, 1, 0],
                                [1, 1, 1, 1]
                        ],
                        [
                                [1, 0, 0, 0],
                                [1, 1, 0, 0],
                                [1, 1, 1, 0],
                                [1, 1, 1, 1]
                        ]
                ]
        )
        
        # create the mask
        mask = util.create_shifted_output_mask(seq)

        # CHECK: the retrieved mask is a ByteTensor
        self.assertIsInstance(mask, torch.ByteTensor)

        # CHECK: the mask has the correct shape
        self.assertEqual((seq.size(0), seq.size(1), seq.size(1)), mask.size())

        # CHECK: the mask contains the correct values
        self.assertEqual(target_mask, mask)
    
    def test_shift_output_sequence(self):
        # create test data
        seq = torch.FloatTensor(
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
        target = torch.FloatTensor(
                [
                        [
                                [0, 0, 0, 0, 0],
                                [1, 2, 3, 4, 5],
                                [11, 22, 33, 44, 55]
                        ],
                        [
                                [0, 0, 0, 0, 0],
                                [6, 7, 8, 9, 0],
                                [66, 77, 88, 99, 00]
                        ]
                ]
        )
        
        # shift the sequences
        shifted_seq = util.shift_output_sequence(seq)
        
        # CHECK: the sequence has been shifted correctly
        self.eps = 1e-9
        self.assertEqual(target, shifted_seq)
