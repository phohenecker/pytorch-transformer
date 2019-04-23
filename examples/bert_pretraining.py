#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""An example of how to pretrain a transformer encoder with BERT."""


import collections
import itertools
import typing

import gensim.models.word2vec as word2vec
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import transformer
import transformer.bert as bert


__author__ = "Patrick Hohenecker"
__copyright__ = (
        "Copyright (c) 2019, Patrick Hohenecker\n"
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
__version__ = "2019.1"
__date__ = "23 Apr 2019"
__maintainer__ = "Patrick Hohenecker"
__email__ = "mail@paho.at"
__status__ = "Development"


# ==================================================================================================================== #
#  C O N S T A N T S                                                                                                   #
# ==================================================================================================================== #


Token = collections.namedtuple("Token", ["index", "word"])
"""This is used to store index-word pairs."""

DATA = [
        "where the streets have no name",
        "we ' re still building then burning down love",
        "burning down love",
        "and when i go there , i go there with you",
        "it ' s all i can do"
]
"""list[str]: The already preprocessed training data."""


#  SPECIAL TOKENS  #####################################################################################################

SOS = Token(0, "<sos>")
"""The start-of-sequence token."""

EOS = Token(1, "<eos>")
"""The end-of-sequence token."""

PAD = Token(2, "<pad>")
"""The padding token."""

MASK = Token(3, "<mask>")
"""The mask token."""


#  MODEL CONFIG  #######################################################################################################

DIMENSIONS = (256, 32, 32)
"""tuple[int]: A tuple of d_model, d_k, d_v."""

DROPOUT_RATE = 0.1
"""float: The used dropout rate."""

EMBEDDING_SIZE = DIMENSIONS[0]
"""int: The used embedding size."""

NUM_LAYERS = 6
"""int: The number of layers in the trained transformer encoder."""


#  TRAINING DETAILS  ###################################################################################################

GPU = False  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< SET THIS TO True, IF YOU ARE USING A MACHINE WITH A GPU!
"""bool: Indicates whether to make use of a GPU."""

LEARNING_RATE = 0.0001
"""float: The used learning rate."""

NUM_EPOCHS = 500
"""int: The total number of training epochs."""

NUM_HEADS = 6
"""int: The number of attention heads to use."""


# ==================================================================================================================== #
#  H E L P E R  F U N C T I O N S                                                                                      #
# ==================================================================================================================== #


def prepare_data() -> typing.Tuple[typing.List[typing.List[str]], collections.OrderedDict]:
    """Preprocesses the training data, and creates the vocabulary.
    
    Returns:
        list[list[str]]: The training data as list of samples, each of which is a list of words.
        collections.OrderedDict: The vocabulary as an ``OrderedDict`` from words to indices.
    """
    
    # gather all words that appear in the data
    all_words = set()
    for sample in DATA:
        all_words.update(sample.split(" "))
    
    # create the vocabulary
    vocab = collections.OrderedDict(
            [
                    (SOS.word, SOS.index),
                    (EOS.word, EOS.index),
                    (PAD.word, PAD.index),
                    (MASK.word, MASK.index)
            ]
    )
    for idx, word in enumerate(sorted(all_words)):
        vocab[word] = idx + 4
    
    # split, add <sos>...<eos>, and pad the dataset
    data = [[SOS.word] + sample.split(" ") + [EOS.word] for sample in DATA]
    max_len = max(len(sample) for sample in data)
    data = [sample + ([PAD.word] * (max_len - len(sample))) for sample in data]
    
    return data, vocab


# ==================================================================================================================== #
#  M A I N                                                                                                             #
# ==================================================================================================================== #


def main():

    # fetch the training data
    data, vocab = prepare_data()

    # create the word embeddings with word2vec and positional embeddings
    emb_model = word2vec.Word2Vec(
            sentences=data,
            size=EMBEDDING_SIZE,
            min_count=1
    )
    for word in vocab.keys():
        if word not in emb_model.wv:
            emb_model.wv[word] = np.zeros((EMBEDDING_SIZE,))
    word_emb_mat = nn.Parameter(
        data=torch.FloatTensor([emb_model[word] for word in vocab.keys()]),
        requires_grad=False
    )
    word_emb = nn.Embedding(len(vocab), EMBEDDING_SIZE)
    word_emb.weight = word_emb_mat
    pos_emb = nn.Embedding(len(data[0]), EMBEDDING_SIZE)
    pos_emb.weight.require_grad = True

    # turn the dataset into a tensor of word indices
    data = torch.LongTensor([[vocab[word] for word in sample] for sample in data])
    
    # create the encoder, the pretraining loss, and the optimizer
    encoder = transformer.Encoder(
            NUM_LAYERS,  # num_layers
            NUM_HEADS,  # num_heads
            *DIMENSIONS,  # dim_model / dim_keys / dim_values
            DROPOUT_RATE,  # residual_dropout
            DROPOUT_RATE,  # attention_dropout
            PAD.index  # pad_index
    )
    loss = bert.MLMLoss(
            encoder,
            word_emb,
            pos_emb,
            MASK.index
    )
    optimizer = optim.Adam(
            itertools.chain(encoder.parameters(), loss.parameters()),
            lr=LEARNING_RATE
    )

    # move to GPU, if possible
    if GPU:
        data = data.cuda()
        encoder.cuda()
        loss.cuda()  # -> also moves embeddings to the GPU

    # pretrain the encoder
    for epoch in range(NUM_EPOCHS):
        
        # compute the loss
        optimizer.zero_grad()
        current_loss = loss(data)
        print("EPOCH", epoch + 1, ": LOSS =", current_loss.item())
        
        # update the model
        current_loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()
