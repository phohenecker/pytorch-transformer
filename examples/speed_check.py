#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A performance comparison of the transformer and a traditional recurrent attention-based model.

This module measures how long it takes to process one training batch of a (random) sequence-to-sequence task. The
architecture of the recurrent model that the transformer is compared with uses additive attention and GRUs for both
encoder and decoder.
"""


import time

import numpy as np
import torch

import transformer

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
__date__ = "Oct 30, 2018"
__maintainer__ = "Patrick Hohenecker"
__email__ = "mail@paho.at"
__status__ = "Development"


BATCH_SIZE = 128
"""int: The size of the generated batch of input sequences."""

EMBEDDING_SIZE = 100
"""int: The size of each token of the input sequences."""

GPU = False  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< SET THIS TO True, IF YOU ARE USING A MACHINE WITH A GPU!
"""bool: Indicates whether to make use of a GPU."""

HIDDEN_SIZE = 300
"""int: The size of any hidden layers."""

INPUT_LEN = 100
"""int: The length of the (randomly) generated input sequence."""

NUM_RUNS = 5
"""int: The total number of times that each model is ran and the according execution time tracked."""

VOCAB_SIZE = 10000
"""int: The size of the used vocabulary."""


# ==================================================================================================================== #
#  R E C U R R E N T  M O D E L                                                                                        #
# ==================================================================================================================== #


class EncDecWithAttn(nn.Module):
    """A very simple attention-based encoder-decoder model.
    
    Here is a schematic of the implemented model:
    
        word-1 \
        word-2 |     +-----------------+    +--------------------+    +----------------+
        word-3  > -> | Encoder (BiGRU) | -> | Decoder (GRU+attn) | -> | linear+softmax | -> distribution over vocabulary
        word-4 |     +-----------------+    +--------------------+    +----------------+
        ...    /
    """
    
    def __init__(self, emb: nn.Embedding, hidden_size: int):
        super().__init__()
        self._emb = emb
        
        # create the decoder
        self._encoder = nn.GRU(
                emb.embedding_dim,  # input_size
                hidden_size,        # hidden_size,
                bidirectional=True
        )
        self._encoder_init = nn.Parameter(torch.FloatTensor(2, hidden_size))
        
        # create the decoder
        self._decoder = nn.GRU(
                2 * emb.embedding_dim + 2 * hidden_size,  # input_size
                hidden_size                               # hidden_size
        )
        self._decoder_init_hidden = nn.Parameter(torch.FloatTensor(1, hidden_size))
        self._decoder_init_output = nn.Parameter(torch.FloatTensor(1, emb.embedding_dim))
        
        # add an additional feed-forward layer to be used on top of the decoder
        self._output_proj = nn.Sequential(
                nn.Linear(hidden_size, emb.num_embeddings),
                nn.Softmax(dim=1)
        )
        
        # create module for computing the attention scores
        self._attn = nn.Sequential(
                nn.Linear(3 * hidden_size, hidden_size, bias=False),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=False),
                nn.Softmax(dim=1)
        )
        
        self.reset_parameters()
    
    def forward(self, input_seq: torch.LongTensor, target_seq) -> torch.FloatTensor:
        # embed + encode input sequence
        input_seq = self._emb(input_seq)
        enc_seq, _ = self._encoder(
                input_seq,
                self._encoder_init.expand(input_seq.size(1), *self._encoder_init.size()).transpose(0, 1).contiguous()
        )

        all_outputs = []  # used the store the outputs for all time steps
        
        # these are used to store the decoder's last state as well as last output produced
        last_hidden = self._decoder_init_hidden \
                .expand(input_seq.size(1), *self._decoder_init_hidden.size()) \
                .transpose(0, 1)
        last_hidden = last_hidden.contiguous()
        last_output = self._decoder_init_output.expand(input_seq.size(1), self._decoder_init_output.size(1))
        last_output = last_output.contiguous()
        
        # iterate over the input sequence token-by-token, and run the decoder
        for idx, token in enumerate(input_seq):
            
            # run attention to compute a glimpse of the encoded input sequence
            attn_scores = torch.cat(
                    [enc_seq, last_hidden.expand(enc_seq.size(0), last_hidden.size(1), last_hidden.size(2))],
                    dim=2
            )
            attn_scores = self._attn(attn_scores)
            glimpse = (enc_seq * attn_scores).sum(dim=0)
            
            # add a 0-th time-dimension to all inputs of the decoder
            token = token.unsqueeze(0)
            glimpse = glimpse.unsqueeze(0)
            last_output = last_output.unsqueeze(0)
            
            # run the decoder + softmax on top
            _, last_hidden = self._decoder(torch.cat([token, glimpse, last_output], dim=2), last_hidden)
            last_output = self._output_proj(last_hidden.squeeze(0))
            all_outputs.append(last_output)
            
            # fill in target output
            last_output = target_seq[idx]
            last_output = self._emb(last_output)
        
        return torch.stack(all_outputs)
    
    def reset_parameters(self):
        """Resets all tunable parameters of the module."""
        self._encoder.reset_parameters()
        self._decoder.reset_parameters()
        self._attn[0].reset_parameters()
        self._attn[2].reset_parameters()
        self._output_proj[0].reset_parameters()
        nn.init.normal_(self._encoder_init, std=0.1)
        nn.init.normal_(self._decoder_init_hidden, std=0.1)
        nn.init.normal_(self._decoder_init_output, std=0.1)
        

# ==================================================================================================================== #
#  M A I N                                                                                                             #
# ==================================================================================================================== #


def main():
    # create an embedding matrix + randomly sample input as well as target sequence
    emb = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
    input_seq = torch.from_numpy(np.random.randint(1, VOCAB_SIZE - 1, (INPUT_LEN, BATCH_SIZE)))
    target_seq = torch.from_numpy(np.random.randint(1, VOCAB_SIZE - 1, (INPUT_LEN, BATCH_SIZE)))
    # -> we assume that index 0 is the <PAD> token
    
    # create the models being compared
    recurrent_model = EncDecWithAttn(emb, HIDDEN_SIZE)
    transformer_model = transformer.Transformer(
            emb,                    # text_emb
            0,                      # pad_index
            emb.num_embeddings,     # output_size
            max_seq_len=INPUT_LEN,
            dim_model=HIDDEN_SIZE,
            num_layers=1
    )
    
    # move everything to the GPU, if possible
    if GPU:
        input_seq = input_seq.cuda()
        target_seq = target_seq.cuda()
        emb.cuda()
        recurrent_model.cuda()
        transformer_model.cuda()
    
    # measure how long it takes the recurrent model to process the data
    print("Testing the recurrent attention-based encoder-decoder model...")
    times = []
    for idx in range(NUM_RUNS):
        start = time.time()
        recurrent_model(input_seq, target_seq)
        times.append(time.time() - start)
        print("Run {} finished in {:.3f}s".format(idx + 1, times[-1]))
    print("Avg. duration: {:.3f}s\n".format(np.mean(times)))
    
    # flip the first two dimensions of the data, as the transformer expects the first dimension to be the batch
    input_seq = input_seq.transpose(0, 1)
    target_seq = target_seq.transpose(0, 1)

    # measure how long it takes the transformer model to process the data
    print("Testing the transformer model...")
    times = []
    for idx in range(NUM_RUNS):
        start = time.time()
        transformer_model(input_seq, target_seq)
        times.append(time.time() - start)
        print("Run {} finished in {:.3f}s".format(idx + 1, times[-1]))
    print("Avg. duration: {:.3f}s".format(np.mean(times)))


if __name__ == "__main__":
    main()
