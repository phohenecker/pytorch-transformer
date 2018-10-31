#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""An implementation of the overfitting test for the Transformer model.

A simple test, which often signifies bugs in the implementation of a model, is the overfitting test. To that end, the
considered model is trained and evaluated on the same tiny dataset, which it should be able to overfit easily.
Therefore, the final model should yield very high probabilities for the desired target values. If this is not the case,
however, then there is probably something wrong with the tested model and/or its implementation.

In this module, we test our implementation of the Transformer model on a super-simple translation task from German to
English. To that end, the considered corpus consists of 5 short and already pre-processed sentences, and is specified in
this file (see below).
"""


import collections
import itertools
import typing

import torch

import transformer

from torch import nn
from torch import optim


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


Token = collections.namedtuple("Token", ["index", "word"])
"""This is used to store index-word pairs."""


# ==================================================================================================================== #
#  C O N S T A N T S                                                                                                   #
# ==================================================================================================================== #


#  PARALLEL DATA  ######################################################################################################

DATA_GERMAN = [
        "Alle warten auf das Licht .",
        "Fürchtet euch , fürchtet euch nicht .",
        "Die Sonne scheint mir aus den Augen .",
        "Sie wird heut ' Nacht nicht untergehen .",
        "Und die Welt zählt laut bis 10 ."
]

DATA_ENGLISH = [
        "Everyone is waiting for the light .",
        "Be afraid , do not be afraid .",
        "The sun is shining out of my eyes .",
        "It will not go down tonight .",
        "And the world counts up to 10 loudly ."
]


#  SPECIAL TOKENS  #####################################################################################################

SOS = Token(0, "<sos>")
"""str: The start-of-sequence token."""

EOS = Token(1, "<eos>")
"""str: The end-of-sequence token."""

PAD = Token(2, "<pad>")
"""str: The padding token."""


#  MODEL CONFIG  #######################################################################################################

EMBEDDING_SIZE = 300
"""int: The used embedding size."""

GPU = False  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< SET THIS TO True, IF YOU ARE USING A MACHINE WITH A GPU!
"""bool: Indicates whether to make use of a GPU."""

NUM_EPOCHS = 200
"""int: The total number of training epochs."""


# ==================================================================================================================== #
#  H E L P E R  F U N C T I O N S                                                                                      #
# ==================================================================================================================== #


def eval_model(model: transformer.Transformer, input_seq: torch.LongTensor, target_seq: torch.LongTensor) -> None:
    """Evaluates the the provided model on the given data, and prints the probabilities of the desired translations.
    
    Args:
        model (:class:`transformer.Transformer`): The model to evaluate.
        input_seq (torch.LongTensor): The input sequences, as (batch-size x max-input-seq-len) tensor.
        target_seq (torch.LongTensor): The target sequences, as (batch-size x max-target-seq-len) tensor.
    """
    probs = transformer.eval_probability(model, input_seq, target_seq, pad_index=PAD.index).detach().numpy().tolist()
    
    print("sample       " + ("{}         " * len(probs)).format(*range(len(probs))))
    print("probability  " + ("{:.6f}  " * len(probs)).format(*probs))


def fetch_vocab() -> typing.Tuple[typing.List[str], typing.Dict[str, int]]:
    """Determines the vocabulary, and provides mappings from indices to words and vice versa.
    
    Returns:
        tuple: A pair of mappings, index-to-word and word-to-index.
    """
    # gather all (lower-cased) words that appear in the data
    all_words = set()
    for sentence in itertools.chain(DATA_GERMAN, DATA_ENGLISH):
        all_words.update(word.lower() for word in sentence.split(" "))
    
    # create mapping from index to word
    idx_to_word = [SOS.word, EOS.word, PAD.word] + list(sorted(all_words))
    
    # create mapping from word to index
    word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}
    
    return idx_to_word, word_to_idx


def prepare_data(word_to_idx: typing.Dict[str, int]) -> typing.Tuple[torch.LongTensor, torch.LongTensor]:
    """Prepares the data as PyTorch ``LongTensor``s.
    
    Args:
        word_to_idx (dict[str, int]): A dictionary that maps words to indices in the vocabulary.
    
    Returns:
        tuple: A pair of ``LongTensor``s, the first representing the input and the second the target sequence.
    """
    # break sentences into word tokens
    german = []
    for sentence in DATA_GERMAN:
        german.append([SOS.word] + sentence.split(" ") + [EOS.word])
    english = []
    for sentence in DATA_ENGLISH:
        english.append([SOS.word] + sentence.split(" ") + [EOS.word])
    
    # pad all sentences to equal length
    len_german = max(len(sentence) for sentence in german)
    for sentence in german:
        sentence.extend([PAD.word] * (len_german - len(sentence)))
    len_english = max(len(sentence) for sentence in english)
    for sentence in english:
        sentence.extend([PAD.word] * (len_english - len(sentence)))
    
    # map words to indices in the vocabulary
    german = [[word_to_idx[word.lower()] for word in sentence] for sentence in german]
    english = [[word_to_idx[word.lower()] for word in sentence] for sentence in english]
    
    # create according LongTensors
    german = torch.LongTensor(german)
    english = torch.LongTensor(english)
    
    return german, english


# ==================================================================================================================== #
#  M A I N                                                                                                             #
# ==================================================================================================================== #


def main():
    # fetch vocabulary + prepare data
    idx_to_word, word_to_idx = fetch_vocab()
    input_seq, target_seq = prepare_data(word_to_idx)
    
    # create embeddings to use
    emb = nn.Embedding(len(idx_to_word), EMBEDDING_SIZE)
    emb.reset_parameters()
    
    # create transformer model
    model = transformer.Transformer(
            emb,
            PAD.index,
            emb.num_embeddings,
            max_seq_len=max(input_seq.size(1), target_seq.size(1))
    )

    # create an optimizer for training the model + a X-entropy loss
    optimizer = optim.Adam((param for param in model.parameters() if param.requires_grad), lr=0.0001)
    loss = nn.CrossEntropyLoss()
    
    print("Initial Probabilities of Translations:")
    print("--------------------------------------")
    eval_model(model, input_seq, target_seq)
    print()
    
    # move model + data on the GPU (if possible)
    if GPU:
        model.cuda()
        input_seq = input_seq.cuda()
        target_seq = target_seq.cuda()

    # train the model
    for epoch in range(NUM_EPOCHS):
        print("training epoch {}...".format(epoch + 1), end=" ")
    
        predictions = model(input_seq, target_seq)
        optimizer.zero_grad()
        current_loss = loss(
                predictions.view(predictions.size(0) * predictions.size(1), predictions.size(2)),
                target_seq.view(-1)
        )
        current_loss.backward()
        optimizer.step()
    
        print("OK (loss: {:.6f})".format(current_loss.item()))
    
    # put model in evaluation mode
    model.eval()

    print()
    print("Final Probabilities of Translations:")
    print("------------------------------------")
    eval_model(model, input_seq, target_seq)
    
    # randomly sample outputs from the input sequences based on the probabilities computed by the trained model
    sampled_output = transformer.sample_output(model, input_seq, EOS.index, PAD.index, target_seq.size(1))

    print()
    print("Sampled Outputs:")
    print("----------------")
    for sample_idx in range(input_seq.size(0)):
        for token_idx in range(input_seq.size(1)):
            print(idx_to_word[input_seq[sample_idx, token_idx].item()], end=" ")
        print(" => ", end=" ")
        for token_idx in range(sampled_output.size(1)):
            print(idx_to_word[sampled_output[sample_idx, token_idx].item()], end=" ")
        print()


if __name__ == "__main__":
    main()
