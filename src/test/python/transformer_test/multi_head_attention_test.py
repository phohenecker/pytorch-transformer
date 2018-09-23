# -*- coding: utf-8 -*-


import numpy as np
import torch
import torchtestcase as ttc

from torch.nn import functional

from transformer import multi_head_attention as mha


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


class MultiHeadAttentionTest(ttc.TorchTestCase):
    
    TOLERANCE = 1e-5
    
    def setUp(self):
        np.seterr(all="raise")
        
        self.num_heads = 2
        self.dim_model = 4
        self.dim_keys = 2
        self.dim_values = 3
        self.batch_size = 2
        self.seq_len = 3

        # create attention mechanism for testing
        self.attn = mha.MultiHeadAttention(self.num_heads, self.dim_model, self.dim_keys, self.dim_values, 0)

        # create dummy data
        self.queries_0 = np.array(
                [
                        [0, 0, 0, 0],
                        [4, 4, 4, 4],
                        [8, 8, 8, 8]
                ],
                dtype=np.float32
        )
        self.keys_0 = 2 * self.queries_0
        self.values_0 = 3 * self.queries_0
        self.queries_1 = np.array(
                [
                        [0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11]
                ],
                dtype=np.float32
        )
        self.keys_1 = 2 * self.queries_1
        self.values_1 = 3 * self.queries_1
        
        # create tensors to provided as input data to the attention mechanism
        self.in_queries = torch.stack(
                [
                        torch.from_numpy(self.queries_0),
                        torch.from_numpy(self.queries_1)
                ]
        )
        self.in_keys = torch.stack(
                [
                        torch.from_numpy(self.keys_0),
                        torch.from_numpy(self.keys_1)
                ]
        )
        self.in_values = torch.stack(
                [
                        torch.from_numpy(self.values_0),
                        torch.from_numpy(self.values_1)
                ]
        )
    
    def assertArrayEqualsTensor(self, a: np.ndarray, t: torch.Tensor):
        if np.abs(a - t.detach().numpy()).max() > self.TOLERANCE:
            raise AssertionError("The values are different!")
    
    def test_apply_attention(self):
        # project queries, keys, and values to the needed dimensions
        in_queries, in_keys, in_values = self.attn._project_inputs(self.in_queries, self.in_keys, self.in_values)

        # CHECK: ensure that inputs have dimensions as expected
        self.assertEqual((self.batch_size, self.num_heads, self.seq_len, self.dim_keys), in_queries.size())
        self.assertEqual((self.batch_size, self.num_heads, self.seq_len, self.dim_keys), in_keys.size())
        self.assertEqual((self.batch_size, self.num_heads, self.seq_len, self.dim_values), in_values.size())
        
        # compute attended values
        attn_values = self.attn._apply_attention(in_queries, in_keys, in_values, None)
        
        # CHECK: the retrieved tensor has the correct shape
        self.assertEqual((self.batch_size, self.num_heads, self.seq_len, self.dim_values), attn_values.size())
        
        for sample_idx in range(self.batch_size):   # iterate over all samples
            for head_idx in range(self.num_heads):  # iterate over all heads
                
                # compute the attention scores for the current head
                attn_scores = torch.matmul(
                        in_queries[sample_idx][head_idx],
                        in_keys[sample_idx][head_idx].transpose(0, 1)
                )
                attn_scores /= np.sqrt(self.dim_keys)
                attn_scores = functional.softmax(attn_scores, dim=1)
                
                # compute attended values for the current head
                target_attn_values = torch.matmul(attn_scores, in_values[sample_idx][head_idx])
                
                # CHECK: the retrieved attention values are correct
                self.assertEqual(target_attn_values, attn_values[sample_idx][head_idx])
        
        # recompute attended values with 1-mask
        attn_values_2 = self.attn._apply_attention(
                in_queries,
                in_keys,
                in_values,
                torch.ones(self.batch_size, self.in_queries.size(1), self.in_keys.size(1)).byte()
        )
        
        # CHECK: providing the mask did not change the attended values
        self.assertEqual(attn_values, attn_values_2)
        
        # create "short" keys/values
        _, short_in_keys, short_in_values = self.attn._project_inputs(
                self.in_queries,
                self.in_keys[:, :2, :],
                self.in_values[:, :2, :]
        )
        
        # compute attended values for the short inputs
        short_attn_values = self.attn._apply_attention(in_queries, short_in_keys, short_in_values, None).detach()
        
        # compute short attended values using a mask rather than short inputs
        short_attn_values_2 = self.attn._apply_attention(
                in_queries,
                in_keys,
                in_values,
                torch.ByteTensor(
                        [
                                [                   # sample 0
                                        [1, 1, 0],  # query 0
                                        [1, 1, 0],  # query 1
                                        [1, 1, 0]   # query 2
                                ],
                                [                   # sample 1
                                        [1, 1, 0],  # query 0
                                        [1, 1, 0],  # query 1
                                        [1, 1, 0]   # query 2
                                ]
                        ]
                )
        ).detach()
        
        # CHECK: attention over short values yielded the same values as using the mask
        self.eps = self.TOLERANCE
        self.assertEqual(short_attn_values[:, 0], short_attn_values_2[:, 0])
        
        # CHECK: if the mask is all 0, then the retrieved values are 0 as well
        self.eps = 0
        self.assertEqual(
                torch.zeros(in_queries.size()),
                self.attn._apply_attention(
                        in_queries,
                        in_keys,
                        in_values,
                        torch.zeros(in_queries.size(0), in_queries.size(1), in_keys.size(1)).byte()
                )
        )
    
    def test_project_inputs(self):
        # fetch projection matrices of the first head
        query_projection_0 = self.attn.query_projection[0].detach().numpy()
        key_projection_0 = self.attn.key_projection[0].detach().numpy()
        value_projection_0 = self.attn.value_projection[0].detach().numpy()
        
        # fetch projection matrices of the second head
        query_projection_1 = self.attn.query_projection[1].detach().numpy()
        key_projection_1 = self.attn.key_projection[1].detach().numpy()
        value_projection_1 = self.attn.value_projection[1].detach().numpy()
        
        # CHECK: ensure that inputs have dimensions as expected
        self.assertEqual((self.batch_size, self.seq_len, self.dim_model), self.in_queries.size())
        self.assertEqual((self.batch_size, self.seq_len, self.dim_model), self.in_keys.size())
        self.assertEqual((self.batch_size, self.seq_len, self.dim_model), self.in_values.size())
        
        # run input projection
        proj_queries, proj_keys, proj_values = self.attn._project_inputs(self.in_queries, self.in_keys, self.in_values)
        
        # CHECK: the projected values have the correct shapes
        self.assertEqual((self.batch_size, self.num_heads, self.seq_len, self.dim_keys), proj_queries.size())
        self.assertEqual((self.batch_size, self.num_heads, self.seq_len, self.dim_keys), proj_keys.size())
        self.assertEqual((self.batch_size, self.num_heads, self.seq_len, self.dim_values), proj_values.size())
        
        # CHECK: queries are projected correctly
        self.assertArrayEqualsTensor(np.matmul(self.queries_0, query_projection_0), proj_queries[0][0])
        self.assertArrayEqualsTensor(np.matmul(self.queries_0, query_projection_1), proj_queries[0][1])
        self.assertArrayEqualsTensor(np.matmul(self.queries_1, query_projection_0), proj_queries[1][0])
        self.assertArrayEqualsTensor(np.matmul(self.queries_1, query_projection_1), proj_queries[1][1])
        
        # CHECK: keys are projected correctly
        self.assertArrayEqualsTensor(np.matmul(self.keys_0, key_projection_0), proj_keys[0][0])
        self.assertArrayEqualsTensor(np.matmul(self.keys_0, key_projection_1), proj_keys[0][1])
        self.assertArrayEqualsTensor(np.matmul(self.keys_1, key_projection_0), proj_keys[1][0])
        self.assertArrayEqualsTensor(np.matmul(self.keys_1, key_projection_1), proj_keys[1][1])
        
        # CHECK: values are projected correctly
        self.assertArrayEqualsTensor(np.matmul(self.values_0, value_projection_0), proj_values[0][0])
        self.assertArrayEqualsTensor(np.matmul(self.values_0, value_projection_1), proj_values[0][1])
        self.assertArrayEqualsTensor(np.matmul(self.values_1, value_projection_0), proj_values[1][0])
        self.assertArrayEqualsTensor(np.matmul(self.values_1, value_projection_1), proj_values[1][1])
    
    def test_project_output(self):
        # fetch projection matrix
        output_projection = self.attn.output_projection
        
        # compute attention values for all queries
        attn_values = self.attn._apply_attention(
                *self.attn._project_inputs(self.in_queries, self.in_keys, self.in_values),
                None
        )

        # CHECK: ensure that attention values have the correct shape
        self.assertEqual((self.batch_size, self.num_heads, self.seq_len, self.dim_values), attn_values.size())
        
        # run output projection
        output = self.attn._project_output(attn_values)
        
        # CHECK: ensure that the output has the expected shape
        self.assertEqual((self.batch_size, self.seq_len, self.dim_model), output.size())

        for sample_idx in range(self.batch_size):  # iterate over all samples
            for query_idx in range(self.seq_len):  # iterate over all queries
            
                # concatenate the values retrieved by the single heads (as row vector)
                concat_values = torch.cat(
                        [
                                attn_values[sample_idx][0][query_idx],
                                attn_values[sample_idx][1][query_idx]
                        ]
                ).unsqueeze(0)
                
                # project concatenated values
                target_output = torch.matmul(concat_values, output_projection).squeeze()
                
                # CHECK: the retrieved output is correct
                self.eps = self.TOLERANCE
                self.assertEqual(target_output, output[sample_idx][query_idx])
