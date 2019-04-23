# -*- coding: utf-8 -*-


import math
import numbers
import random

import insanity
import numpy as np
import torch
import torch.nn as nn

import transformer.encoder as encoder
import transformer.util as util


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


class MLMLoss(nn.Module):
    """The masked language-model (MLM) loss function for pretraining a transformer encoder.
    
    Unlike other loss functions, an ``MLMLoss`` has trainable parameters, which are part of a linear layer with a
    softmax on top (cf. :attr:`output_layer`) that is used for predicting masked/obliterated tokens. These have to be
    optimized together with the parameters of the pretrained encoder.
    """
    
    def __init__(
            self,
            model: encoder.Encoder,
            word_emb: nn.Embedding,
            pos_emb: nn.Embedding,
            mask_index: int,
            prediction_rate: numbers.Real = 0.15,
            mask_rate: numbers.Real = 0.8,
            random_rate: numbers.Real = 0.1
    ):
        """Creates a new instance of ``BERTLoss`.

        Args:
            model (encoder.Encoder): The encoder model being pretrained.
            word_emb (nn.Embedding): The used word embeddings.
            pos_emb (nn.Embedding): The used positional embeddings.
            mask_index (int): The index of the mask token.
            prediction_rate (numbers.Real, optional): The percentage of tokens in each training sequence that
                predictions are computed for, which is set to ``0.8``, by default.
            mask_rate (numbers.Real, optional): Among all tokens that predictions are computed for, the percentage of
                tokens that are replaced with the mask token, as specified by ``mask_index``. This is set to ``0.8``, by
                default.
            random_rate (numbers.Real, optional): Among all tokens that predictions are computed for, the percentage of
                tokens that are randomly replaced with other tokens. This is set to ``0.1``, by default.
        """
        super().__init__()
        
        # sanitize args
        insanity.sanitize_type("model", model, encoder.Encoder)
        insanity.sanitize_type("word_emb", word_emb, nn.Embedding)
        insanity.sanitize_type("pos_emb", word_emb, nn.Embedding)
        if pos_emb.embedding_dim != word_emb.embedding_dim:
            raise ValueError("<pos_emb> is not compatible with <word_emb>!")
        insanity.sanitize_type("mask_index", mask_index, int)
        if mask_index < 0 or mask_index >= word_emb.num_embeddings:
            raise ValueError("The <mask_index> does not exist in <word_emb>!")
        insanity.sanitize_type("prediction_rate", prediction_rate, numbers.Real)
        prediction_rate = float(prediction_rate)
        insanity.sanitize_range("prediction_rate", prediction_rate, minimum=0, maximum=1)
        insanity.sanitize_type("mask_rate", mask_rate, numbers.Real)
        mask_rate = float(mask_rate)
        insanity.sanitize_range("mask_rate", mask_rate, minimum=0, maximum=1)
        insanity.sanitize_type("random_rate", random_rate, numbers.Real)
        random_rate = float(random_rate)
        insanity.sanitize_range("random_rate", random_rate, minimum=0, maximum=1)
        if mask_rate + random_rate > 1:
            raise ValueError("<mask_rate> + <random_rate> has to be at most 1!")
        
        # store args
        self._mask_index = mask_index
        self._mask_rate = mask_rate
        self._model = model
        self._pad_index = model.pad_index
        self._pos_emb = pos_emb
        self._prediction_rate = prediction_rate
        self._random_rate = random_rate
        self._word_emb = word_emb
        
        # create an output layer, which is trained together with the model, for predicting masked tokens
        self._output_layer = nn.Sequential(
                nn.Linear(self._word_emb.embedding_dim, self._word_emb.num_embeddings),
                nn.Softmax(dim=1)
        )
        
        # create the used loss function
        self._loss = nn.CrossEntropyLoss()
    
    #  PROPERTIES  #####################################################################################################
    
    @property
    def output_layer(self) -> nn.Sequential:
        """nn.Sequential: A linear layer with a softmax on top, which is used for predicting masked/obliterated tokens.
        """
        return self._output_layer
    
    #  METHODS  ########################################################################################################
    
    def forward(self, batch: torch.LongTensor) -> torch.FloatTensor:
        """Computes the loss function.

        Args:
            batch (torch.LongTensor): A batch of training data, as (batch-size x max-seq-len)-tensor.

        Returns:
            torch.FloatTensor: The computed loss.
        """
        # sanitize args
        insanity.sanitize_type("batch", batch, torch.Tensor)
        if batch.dtype != torch.int64:
            raise TypeError("<batch> has to be a LongTensor!")
        if batch.dim() != 2:
            raise ValueError("<batch> has to be a 2d tensor!")
        
        # create the padding mask to use
        padding_mask = util.create_padding_mask(batch, self._pad_index)
        
        # create a tensor of indices, which is used to retrieve the according positional embeddings below
        index_seq = batch.new(range(batch.size(1))).unsqueeze(0).expand(batch.size(0), -1)
        
        # compute the sequence lengths for all samples in the batch
        seq_len = (batch != self._pad_index).sum(dim=1).cpu().numpy().tolist()
        
        # randomly choose the tokens to compute predictions for
        pred_mask = padding_mask.new(*batch.size()).zero_().long()  # all tokens being predicted
        mask_mask = padding_mask.new(*batch.size()).zero_().long()  # token replaced with <MASK>
        random_mask = padding_mask.new(*batch.size()).zero_().long()  # tokens replace with random tokens
        for sample_idx, sample_len in enumerate(seq_len):  # iterate over all samples in the batch
            
            # determine how many tokens to computed predictions for
            num_pred = int(math.ceil(sample_len * self._prediction_rate))  # num of tokens predictions are computed for
            num_mask = int(math.floor(num_pred * self._mask_rate))  # num of tokens replaced with <MASK>
            num_random = int(math.ceil(num_pred * self._random_rate))  # num of tokens randomly replaced
            
            # randomly select indices to compute predictions for
            pred_indices = list(range(sample_len))
            random.shuffle(pred_indices)
            pred_indices = pred_indices[:num_pred]
            
            # prepare the <MASK>-mask
            for token_idx in pred_indices[:num_mask]:
                pred_mask[sample_idx, token_idx] = 1
                mask_mask[sample_idx, token_idx] = 1
            
            # prepare the random-mask
            for token_idx in pred_indices[num_mask:(num_mask + num_random)]:
                pred_mask[sample_idx, token_idx] = 1
                random_mask[sample_idx, token_idx] = 1
            
            # remaining tokens that predictions are computed for are left untouched
            for token_idx in pred_indices[(num_mask + num_random):]:
                pred_mask[sample_idx, token_idx] = 1
        
        # replace predicted tokens in the batch appropriately
        masked_batch = (
                batch * (1 - mask_mask) * (1 - random_mask) +
                mask_mask * batch.new(*batch.size()).fill_(self._mask_index) +
                random_mask * (batch.new(*batch.size()).double().uniform_() * self._word_emb.num_embeddings).long()
        )
        
        # embed the batch
        masked_batch = self._word_emb(masked_batch) + self._pos_emb(index_seq)
        
        # encode sequence in the batch using BERT
        enc = self._model(masked_batch, padding_mask)
        
        # turn encodings, the target token indices (that we seek to predict), and the prediction mask, into matrices,
        # such that each row corresponds with one token
        enc = enc.view(enc.size(0) * enc.size(1), enc.size(2))
        target = batch.view(-1)
        pred_mask = pred_mask.view(-1)
        
        # turn the prediction mask into a tensor of indices (to select below)
        pred_mask = pred_mask.new(np.where(pred_mask.detach().cpu().numpy())[0])
        
        # fetch embeddings and target values of those tokens that are being predicted
        enc = enc.index_select(0, pred_mask)
        target = target.index_select(0, pred_mask)
        
        # compute predictions for each encoded token + the according loss
        pred = self._output_layer(enc)
        loss = self._loss(pred, target)
        
        return loss

    def reset_parameters(self) -> None:
        """Resets the loss' tunable parameters that are being trained to predict masked/obliterated tokens.
        
        Notice that this function does **not** reset the used embeddings.
        """
        self._output_layer[0].reset_parameters()
