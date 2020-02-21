"""
Module files as torch.nn.Module subclasses for Seq2seqAgent.
"""

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

from parlai.utils.torch import NEAR_INF
from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai_internal.agents.hred.modules import * 
from parlai_internal.agents.hred.modules import _transpose_hidden_state


class HRED(TorchGeneratorModel):
    """
    The basic structure of the HRED model.
    Defines encoder -> context -> decoder flow.
    """

    RNN_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

    def __init__(
        self,
        num_features,
        contextsize,
        embeddingsize,
        hiddensize,
        numlayers=2,
        dropout=0,
        bidirectional=False,
        rnn_class='lstm',
        lookuptable='unique',
        decoder='same',
        numsoftmax=1,
        attention='none',
        attention_length=48,
        attention_time='post',
        padding_idx=0,
        start_idx=1,
        end_idx=2,
        unknown_idx=3,
        input_dropout=0,
        longest_label=1,
    ):
        # initialize model (adapted from seq2seq)
        super().__init__(
            padding_idx=padding_idx,
            start_idx=start_idx,
            end_idx=end_idx,
            unknown_idx=unknown_idx,
            input_dropout=input_dropout,
            longest_label=longest_label,
        )
        self.attn_type = attention

        rnn_class = HRED.RNN_OPTS[rnn_class]
        self.decoder = HREDRNNDecoder(
            num_features,
            embeddingsize,
            hiddensize,
            padding_idx=padding_idx,
            rnn_class=rnn_class,
            numlayers=numlayers,
            dropout=dropout,
            attn_type=attention,
            attn_length=attention_length,
            attn_time=attention_time,
            bidir_input=bidirectional,
        )

        shared_lt = (
            self.decoder.lt  # share embeddings between rnns
            if lookuptable in ('enc_dec', 'all')
            else None
        )
        shared_rnn = self.decoder.rnn if decoder == 'shared' else None
        self.encoder = RNNEncoder(
            num_features,
            embeddingsize,
            hiddensize,
            padding_idx=padding_idx,
            rnn_class=rnn_class,
            numlayers=numlayers,
            dropout=dropout,
            bidirectional=bidirectional,
            shared_lt=shared_lt,
            shared_rnn=shared_rnn,
            unknown_idx=unknown_idx,
            input_dropout=input_dropout,
        )

        shared_weight = (
            self.decoder.lt  # use embeddings for projection
            if lookuptable in ('dec_out', 'all')
            else None
        )
        self.output = OutputLayer(
            num_features,
            embeddingsize,
            hiddensize,
            dropout=dropout,
            numsoftmax=numsoftmax,
            shared_weight=shared_weight,
            padding_idx=padding_idx,
        )

        # decided to make this part of HRED class instead of its own ContextLayer class
        self.context_layer = nn.GRUCell(hiddensize, contextsize)
        self.csz = contextsize


    def forward(self, xs0, xs1, ys1, ys2):
        # need to modify seq2seq to handle *triples*, not just pairs
        bsz, __ = xs0.shape
        context_vector = torch.zeros((bsz, self.csz))
        encoder_output_0, encoder_state_0, encoder_attn_0 = self.encoder(xs0)
        combined_state = encoder_state_0[0][:,0,:]
    
        # context propagates through time (unlike seq2seq)
        context_vector = self.context_layer(combined_state, context_vector)
        scores_1, preds_1 = self.decode_forced((encoder_output_0, encoder_state_0, encoder_attn_0), 
                                                ys1, 
                                                context_vector)
        encoder_output_1, encoder_state_1, encoder_attn_1 = self.encoder(xs1)
        combined_state = encoder_state_1[0][:,0,:]
        context_vector = self.context_layer(combined_state, context_vector)
        scores_2, preds_2 = self.decode_forced((encoder_output_0, encoder_state_0, encoder_attn_0),
                                                ys2, 
                                                context_vector)
        
        return scores_1, preds_1, scores_2, preds_2, encoder_state_0, encoder_state_1 

    def decode_forced(self, encoder_out, ys, context_vector): 
        encoder_hidden, encoder_state, encoder_attn = encoder_out 
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self.START.detach().expand(bsz, 1), inputs], 1) 

        latent, _ = self.decoder(inputs, encoder_out, context_vector)
        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds
    

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder encoder states according to a new set of indices.
        """
        enc_out, hidden, attn_mask = encoder_states

        # make sure we swap the hidden state around, apropos multigpu settings
        hidden = _transpose_hidden_state(hidden)

        # LSTM or GRU/RNN hidden state?
        if isinstance(hidden, torch.Tensor):
            hid, cell = hidden, None
        else:
            hid, cell = hidden

        if not torch.is_tensor(indices):
            # cast indices to a tensor if needed
            indices = torch.LongTensor(indices).to(hid.device)

        hid = hid.index_select(1, indices)
        if cell is None:
            hidden = hid
        else:
            cell = cell.index_select(1, indices)
            hidden = (hid, cell)

        if self.attn_type != 'none':
            enc_out = enc_out.index_select(0, indices)
            attn_mask = attn_mask.index_select(0, indices)

        # and bring it back to multigpu friendliness
        hidden = _transpose_hidden_state(hidden)

        return enc_out, hidden, attn_mask

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        if torch.is_tensor(incremental_state):
            # gru or vanilla rnn
            return torch.index_select(incremental_state, 0, inds).contiguous()
        elif isinstance(incremental_state, tuple):
            return tuple(
                self.reorder_decoder_incremental_state(x, inds)
                for x in incremental_state
            )

