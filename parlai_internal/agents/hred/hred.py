#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math

from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.core.metrics import SumMetric, AverageMetric, BleuMetric, FairseqBleuMetric
from parlai.utils.misc import warn_once
from parlai.core.torch_agent import  Batch, Output
from parlai.core.message import Message
from parlai.utils.misc import AttrDict, warn_once
from parlai.utils.distributed import is_primary_worker

from .modules import  opt_to_kwargs 
from .model import  HRED

import torch
import torch.nn as nn
import torch.nn.functional as F

class HREDOutput(AttrDict):
    def __init__(self, text_0, text_1, text_0_candidates, text_1_candidates, **kwargs):
        super().__init__(text_0 = text_0, text_1 = text_1, 
                        text_0_candidates = text_0_candidates, 
                        text_1_candidates = text_1_candidates, 
                        **kwargs)

class PPLMetric(AverageMetric):
    def value(self):
        return math.exp(super().value())

class HredBatch(AttrDict):
    """
    Batch is a namedtuple containing data being sent to an agent.

    This is the input type of the train_step and eval_step functions.
    Agents can override the batchify function to return an extended namedtuple
    with additional fields if they would like, though we recommend calling the
    parent function to set up these fields as a base.

    :param text_vec:
        bsz x seqlen tensor containing the parsed text data.

    :param text_lengths:
        list of length bsz containing the lengths of the text in same order as
        text_vec; necessary for pack_padded_sequence.

    :param label_vec:
        bsz x seqlen tensor containing the parsed label (one per batch row).

    :param label_lengths:
        list of length bsz containing the lengths of the labels in same order as
        label_vec.

    :param labels:
        list of length bsz containing the selected label for each batch row (some
        datasets have multiple labels per input example).

    :param valid_indices:
        list of length bsz containing the original indices of each example in the
        batch. we use these to map predictions back to their proper row, since e.g.
        we may sort examples by their length or some examples may be invalid.

    :param candidates:
        list of lists of text. outer list has size bsz, inner lists vary in size
        based on the number of candidates for each row in the batch.

    :param candidate_vecs:
        list of lists of tensors. outer list has size bsz, inner lists vary in size
        based on the number of candidates for each row in the batch.

    :param image:
        list of image features in the format specified by the --image-mode arg.

    :param observations:
        the original observations in the batched order
    """

    def __init__(
        self,
        text_0_vec=None,
        text_0_lengths=None,
        text_1_vec=None,
        text_1_lengths=None,
        labels_1_vec_1=None, 
        labels_1_lengths=None, 
        labels_2_vec_2=None, 
        labels_2_lengths=None, 
        valid_indices=None,
        candidates=None,
        candidate_vecs=None,
        image=None,
        observations=None,
        **kwargs,
    ):
        super().__init__(
            text_0_vec=text_0_vec,
            text_0_lengths=text_0_lengths,
            text_1_vec=text_1_vec,
            text_1_lengths=text_1_lengths,
            labels_1_vec_1=labels_1_vec_1,
            labels_1_lengths=labels_1_lengths,
            labels_2_vec_2=labels_2_vec_2,
            labels_2_lengths=labels_2_lengths,
            valid_indices=valid_indices,
            candidates=candidates,
            candidate_vecs=candidate_vecs,
            image=image,
            observations=observations,
            **kwargs,
        )


class HredAgent(TorchGeneratorAgent):
    """
    Agent which takes an input sequence and produces an output sequence.

    This model supports encoding the input and decoding the output via one of
    several flavors of RNN. It then uses a linear layer (whose weights can
    be shared with the embedding layer) to convert RNN output states into
    output tokens. This model supports greedy decoding, selecting the
    highest probability token at each time step, as well as beam
    search.

    For more information, see the following papers:

    - Neural Machine Translation by Jointly Learning to Align and Translate
      `(Bahdanau et al. 2014) <arxiv.org/abs/1409.0473>`_
    - Sequence to Sequence Learning with Neural Networks
      `(Sutskever et al. 2014) <arxiv.org/abs/1409.3215>`_
    - Effective Approaches to Attention-based Neural Machine Translation
      `(Luong et al. 2015) <arxiv.org/abs/1508.04025>`_
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        agent = argparser.add_argument_group('Seq2Seq Arguments')
        agent.add_argument(
            '-hs',
            '--hiddensize',
            type=int,
            default=128,
            help='size of the hidden layers',
        )
        agent.add_argument(
            '-esz',
            '--embeddingsize',
            type=int,
            default=128,
            help='size of the token embeddings',
        )
        agent.add_argument(
            '-csz',
            '--contextsize',
            type=int,
            default=1000,
            help='size of the context vector',
        )
        agent.add_argument(
            '-nl', '--numlayers', type=int, default=2, help='number of hidden layers'
        )
        agent.add_argument(
            '-dr', '--dropout', type=float, default=0.1, help='dropout rate'
        )
        agent.add_argument(
            '-bi',
            '--bidirectional',
            type='bool',
            default=False,
            help='whether to encode the context with a ' 'bidirectional rnn',
        )
        agent.add_argument(
            '-att',
            '--attention',
            default='none',
            choices=['none', 'concat', 'general', 'dot', 'local'],
            help='Choices: none, concat, general, local. '
            'If set local, also set attention-length. '
            '(see arxiv.org/abs/1508.04025)',
        )
        agent.add_argument(
            '-attl',
            '--attention-length',
            default=48,
            type=int,
            help='Length of local attention.',
        )
        agent.add_argument(
            '--attention-time',
            default='post',
            choices=['pre', 'post'],
            help='Whether to apply attention before or after ' 'decoding.',
        )
        agent.add_argument(
            '-rnn',
            '--rnn-class',
            default='lstm',
            choices=HRED.RNN_OPTS.keys(),
            help='Choose between different types of RNNs.',
        )
        agent.add_argument(
            '-dec',
            '--decoder',
            default='same',
            choices=['same', 'shared'],
            help='Choose between different decoder modules. '
            'Default "same" uses same class as encoder, '
            'while "shared" also uses the same weights. '
            'Note that shared disabled some encoder '
            'options--in particular, bidirectionality.',
        )
        agent.add_argument(
            '-lt',
            '--lookuptable',
            default='unique',
            choices=['unique', 'enc_dec', 'dec_out', 'all'],
            help='The encoder, decoder, and output modules can '
            'share weights, or not. '
            'Unique has independent embeddings for each. '
            'Enc_dec shares the embedding for the encoder '
            'and decoder. '
            'Dec_out shares decoder embedding and output '
            'weights. '
            'All shares all three weights.',
        )
        agent.add_argument(
            '-soft',
            '--numsoftmax',
            default=1,
            type=int,
            help='default 1, if greater then uses mixture of '
            'softmax (see arxiv.org/abs/1711.03953).',
        )
        agent.add_argument(
            '-idr',
            '--input-dropout',
            type=float,
            default=0.0,
            help='Probability of replacing tokens with UNK in training.',
        )

        super(HredAgent, cls).add_cmdline_args(argparser)
        return agent

    @staticmethod
    def model_version():
        """
        Return current version of this model, counting up from 0.

        Models may not be backwards-compatible with older versions. Version 1 split from
        version 0 on Aug 29, 2018. Version 2 split from version 1 on Nov 13, 2018 To use
        version 0, use --model legacy:seq2seq:0 To use version 1, use --model
        legacy:seq2seq:1 (legacy agent code is located in parlai/agents/legacy_agents).
        """
        return 2

    def __init__(self, opt, shared=None):
        """
        Set up model.
        """
        super().__init__(opt, shared)
        self.id = 'Hred'

    def build_model(self, states=None):
        """
        Initialize model, override to change model setup.
        """
        opt = self.opt
        if not states:
            states = {}

        kwargs = opt_to_kwargs(opt)
        model = HRED(
            len(self.dict),
            opt['contextsize'],
            opt['embeddingsize'],
            opt['hiddensize'],
            padding_idx=self.NULL_IDX,
            start_idx=self.START_IDX,
            end_idx=self.END_IDX,
            unknown_idx=self.dict[self.dict.unk_token],
            longest_label=states.get('longest_label', 1),
            **kwargs,
        )

        if opt.get('dict_tokenizer') == 'bpe' and opt['embedding_type'] != 'random':
            print('skipping preinitialization of embeddings for bpe')
        elif not states and opt['embedding_type'] != 'random':
            # `not states`: only set up embeddings if not loading model
            self._copy_embeddings(model.decoder.lt.weight, opt['embedding_type'])
            if opt['lookuptable'] in ['unique', 'dec_out']:
                # also set encoder lt, since it's not shared
                self._copy_embeddings(
                    model.encoder.lt.weight, opt['embedding_type'], log=False
                )

        if states:
            # set loaded states if applicable
            model.load_state_dict(states['model'])

        if opt['embedding_type'].endswith('fixed'):
            print('HRED: fixing embedding weights.')
            model.decoder.lt.weight.requires_grad = False
            model.encoder.lt.weight.requires_grad = False
            if opt['lookuptable'] in ['dec_out', 'all']:
                model.output.weight.requires_grad = False

        return model

    def build_criterion(self):
        # set up criteria
        if self.opt.get('numsoftmax', 1) > 1:
            return nn.NLLLoss(ignore_index=self.NULL_IDX, reduction='sum')
        else:
            return nn.CrossEntropyLoss(ignore_index=self.NULL_IDX, reduction='sum')

    def _set_label_vec(self, obs, add_start, add_end, truncate, index):
        """
        Set the 'labels_vec' field in the observation.

        Useful to override to change vectorization behavior
        """
        #print(obs)
        # convert 'labels' or 'eval_labels' into vectors
        if f'labels_{index}' in obs:
            label_type = f'labels_{index}'
        elif 'eval_labels_{index}' in obs:
            label_type = f'eval_labels_{index}'
        else:
            label_type = None

        if label_type is None:
            return

        elif label_type + f'_vec_{index}' in obs:
            # check truncation of pre-computed vector
            truncated_vec = self._check_truncate(obs[label_type + f'_vec_{index}'], truncate)
            obs.force_set(label_type + f'_vec_{index}', torch.LongTensor(truncated_vec))
        else:
            # pick one label if there are multiple
            lbls = obs[label_type]
            label = lbls[0] if len(lbls) == 1 else self.random.choice(lbls)
            vec_label = self._vectorize_text(label, add_start, add_end, truncate, False)
            obs[label_type + f'_vec_{index}'] = vec_label
            obs[label_type + f'_choice_{index}'] = label

        return obs


    def _set_text_vec(self, obs, history, truncate, index):
        """
        Override the _set_text_vec fxn. Set the 'text_vec' field in the observation.

        Useful to override to change vectorization behavior
        """

        history_string = history.get_history_str()

        if f'text_{index}' not in obs:
            return obs

        if 'text_vec_{index}' not in obs:
            history_string = history.get_history_str()
            #print(f"got history string which is {history_string}") 
            #if history_string is None:
            #    print(f"history is none index {index} in obs {obs}, returning") 
            obs[f'text_vec_{index}'] =  history.parse(obs[f'text_{index}'])
                #return obs

        # check truncation
        if obs.get(f'text_vec_{index}') is not None:
            truncated_vec = self._check_truncate(obs[f'text_vec_{index}'], truncate, True)
            obs.force_set(f'text_vec_{index}', torch.LongTensor(truncated_vec))
        return obs

    def _set_label_cands_vec(self, obs, add_start, add_end, truncate, index):
        """
        Set the 'label_candidates_vec' field in the observation.

        Useful to override to change vectorization behavior
        """
        if 'label_candidates_vecs' in obs:
            if truncate is not None:
                # check truncation of pre-computed vectors
                vecs = obs[f'label_candidates_vecs_{index}']
                for i, c in enumerate(vecs):
                    vecs[i] = self._check_truncate(c, truncate)
        elif self.rank_candidates and obs.get(f'label_candidates_{index}'):
            obs.force_set(f'label_candidates_{index}', list(obs[f'label_candidates_{index}']))
            obs[f'label_candidates_vecs_{index}'] = [
                self._vectorize_text(c, add_start, add_end, truncate, False)
                for c in obs[f'label_candidates_{index}']
            ]
        return obs

    def vectorize(
                  self,
                  obs,
                  history,
                  add_start=True,
                  add_end=True,
                  text_truncate=None,
                  label_truncate=None):
        """
        override core.torch_agent.vectorize 
        to add an extra utterance to the batch
        """

        self._set_text_vec(obs, history, text_truncate, 0)
        self._set_text_vec(obs, history, text_truncate, 1)
        self._set_label_vec(obs, add_start, add_end, label_truncate, 1)
        self._set_label_vec(obs, add_start, add_end, label_truncate, 2)
        self._set_label_cands_vec(obs, add_start, add_end, label_truncate, 1)
        self._set_label_cands_vec(obs, add_start, add_end, label_truncate, 2)
        return obs


    def batchify(self, *args, **kwargs):
        """
        Override batchify options for seq2seq.
        """
        obs_batch = args[0]
        sort = kwargs.get('sort', False)

        if len(obs_batch) == 0:
            return HredBatch()

        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if self.is_valid(ex)]

        if len(valid_obs) == 0:
            return HredBatch()

        valid_inds, exs = zip(*valid_obs)

        # TEXT
        xs_1, x_1_lens = None, None
        xs_2, x_2_lens = None, None

        if any(ex.get(f'text_vec_0') is not None for ex in exs):
            _xs_1 = [ex.get(f'text_vec_0', self.EMPTY) for ex in exs]
            xs_1, x_1_lens = self._pad_tensor(_xs_1)
            # TODO HRED: Implement this later doesn't matter rn
            if sort:
                sort = False  # now we won't sort on labels
                xs, x_lens, valid_inds, exs = argsort(
                    x_lens, xs, x_lens, valid_inds, exs, descending=True
                )

        if any(ex.get(f'text_vec_1') is not None for ex in exs):
            _xs_2 = [ex.get(f'text_vec_1', self.EMPTY) for ex in exs]
            xs_2, x_2_lens = self._pad_tensor(_xs_2)

        # LABELS
        labels_1_avail = any('labels_1_vec_1' in ex for ex in exs)
        some_labels_1_avail = labels_1_avail or any('eval_labels_1_vec_1' in ex for ex in exs)

        ys_1, y_1_lens, labels_1 = None, None, None
        if some_labels_1_avail:
            field = 'labels_1' if labels_1_avail else 'eval_labels_1'

            label_1_vecs = [ex.get(field + '_vec_1', self.EMPTY) for ex in exs]
            labels_1 = [ex.get(field + '_choice_1') for ex in exs]
            y_lens_1 = [y.shape[0] for y in label_1_vecs]

            ys_1, y_1_lens = self._pad_tensor(label_1_vecs)

            if sort and xs is None:
                ys, valid_inds, label_vecs, labels, y_lens = argsort(
                    y_lens, ys, valid_inds, label_vecs, labels, y_lens, descending=True
                )

        labels_2_avail = any('labels_2_vec_2' in ex for ex in exs)
        some_labels_2_avail = labels_2_avail or any('eval_labels_2_vec_2' in ex for ex in exs)

        ys_2, y_2_lens, labels_2 = None, None, None
        if some_labels_2_avail:
            field = 'labels_2' if labels_2_avail else 'eval_labels_2'

            label_2_vecs = [ex.get(field + '_vec_2', self.EMPTY) for ex in exs]
            labels_2 = [ex.get(field + '_choice_2') for ex in exs]
            y_lens_2 = [y.shape[0] for y in label_2_vecs]

            ys_2, y_2_lens = self._pad_tensor(label_2_vecs)

        # LABEL_CANDIDATES
        cands, cand_vecs = None, None
        if any('label_candidates_vecs' in ex for ex in exs):
            cands = [ex.get('label_candidates', None) for ex in exs]
            cand_vecs = [ex.get('label_candidates_vecs', None) for ex in exs]

        # IMAGE
        imgs = None
        if any('image' in ex for ex in exs):
            imgs = [ex.get('image', None) for ex in exs]

        return HredBatch(
            text_0_vec=xs_1,
            text_0_lengths=x_1_lens,
            text_1_vec=xs_2,
            text_1_lengths=x_2_lens,
            label_1_vec=ys_1,
            label_1_lengths=y_1_lens,
            label_2_vec=ys_2,
            label_2_lengths=y_2_lens,
            labels_1=labels_1,
            labels_2=labels_2,
            valid_indices=valid_inds,
            candidates=cands,
            candidate_vecs=cand_vecs,
            image=imgs,
            observations=exs,
        )

    def batch_act(self, observations):
        """
        Process a batch of observations (batchsize list of message dicts).

        These observations have been preprocessed by the observe method.

        Subclasses can override this for special functionality, but if the
        default behaviors are fine then just override the ``train_step`` and
        ``eval_step`` methods instead. The former is called when labels are
        present in the observations batch; otherwise, the latter is called.
        """
        # clear local metrics before anything else
        self._local_metrics.clear()

        # initialize a list of replies with this agent's id
        batch_reply = [
            Message({'id': self.getID(), 'episode_done': False}) for _ in observations
        ]

        # check if there are any labels available, if so we will train on them
        self.is_training = any('labels_1' in obs for obs in observations)

        # create a batch from the vectors
        batch = self.batchify(observations)
        if (
            'label_1_vec' in batch
            and 'text_0_vec' in batch
            and batch.label_1_vec is not None
            and batch.text_1_vec is not None
        ):
            # tokens per batch
            # we divide by the binary is_primary_worker() so that the numerator is
            # num_tokens in all workers, and the denominator is 1.
            # TODO HRED: add to this
            tbp = AverageMetric(
                (batch.label_1_vec != self.NULL_IDX).sum().item()
                + (batch.text_0_vec != self.NULL_IDX).sum().item(),
                float(is_primary_worker()),
            )
            self.global_metrics.add('tokens_per_batch', tbp)

        if self.is_training:
            output = self.train_step(batch)
        else:
            with torch.no_grad():
                # save memory and compute by disabling autograd.
                # use `with torch.enable_grad()` to gain back gradients.
                output = self.eval_step(batch)
                #print(f"output of eval is {output}")

        if output is not None:
            # local metrics are automatically matched up
            self.match_batch(batch_reply, batch.valid_indices, output)

        # broadcast the metrics back
        for k, values in self._local_metrics.items():
            if len(values) != len(batch.valid_indices):
                raise IndexError(
                    f"Batchsize mismatch on metric {k} (got {len(values)}, "
                    f"expected {len(batch.valid_indices)}"
                )
            for i, value in zip(batch.valid_indices, values):
                if 'metrics' not in batch_reply[i]:
                    batch_reply[i]['metrics'] = {}
                batch_reply[i]['metrics'][k] = value

        # Make sure we push all the metrics to main thread in hogwild/workers
        self.global_metrics.flush()
        #print(f"batch reply is {batch_reply}")
        return batch_reply

    def observe(self, observation):
        """
        Process incoming message in preparation for producing a response.

        This includes remembering the past history of the conversation.
        """
        # TODO: Migration plan: TorchAgent currently supports being passed
        # observations as vanilla dicts for legacy interop; eventually we
        # want to remove this behavior and demand that teachers return Messages
        #print(f"history is {self.history.history_strings}") 
        #if len(self.history.history_strings) > 0:
        #    observation.force_set("text_0", self.history.history_strings[-1])
        #else:
        #    observation.force_set("text_0", "__SILENCE__") 
        observation['text_0'] = observation.get('text0')
        #print(f"observing {observation}") 
        # NEXT 2 LINES ARE DIRTY HACKS
        if 'labels' in observation:
            observation.force_set('labels', observation.get('labels_1'))
        else:
            observation['labels'] = observation.get('labels_1')
        #observation['eval_labels'] = observation.get('eval_labels_1')
        if 'eval_labels' not in observation:
            observation['eval_labels'] = observation.get('eval_labels_1')
        if "text_1" in observation:
            observation.force_set('text',  observation.get('text_1')) 
            observation.force_set('text_1',  observation.get('text_1')) 
        else:
            observation.force_set('text', observation.get('text') ) 
            observation.force_set('text_1', observation.get('text') ) 
        return super().observe(observation)

    def self_observe(self, self_message: Message) -> None:
        """
        Observe one's own utterance.

        This is used so that the agent can incorporate its own response into
        the dialogue history after a batch_act. Failure to implement this will
        result in an agent that cannot hear itself speak.

        :param self_message:
            The message corresponding to the output from batch_act.
        """
        #print(f"self observe {self_message}") 
        use_reply = self.opt.get('use_reply', 'label')

        # quick check everything is in order
        self._validate_self_observe_invariants()

        assert self.observation is not None
        if self.observation['episode_done']:
            # oh this was the last example in the episode. reset the history
            self.history.reset()
            # additionally mark the last observation as invalid
            self.observation = None
            # and clear the safety check
            self.__expecting_clear_history = False
            return

        # We did reply! Safety check is good next round.
        self.__expecting_to_reply = False

        # actually ingest the label
        if use_reply == 'none':
            # we're not including our own responses anyway.
            return
        elif use_reply == 'label':
            # first look for the true label
            label_key = (
                'labels_1'
                if 'labels_1' in self.observation
                else 'eval_labels_1'
                if 'eval_labels_1' in self.observation
                else None
            )
            if label_key is not None:
                lbls = self.observation[label_key]
                last_reply = lbls[0] if len(lbls) == 1 else self.random.choice(lbls)
                self.history.add_reply(last_reply)
                return
            # you might expect a hard failure here, but in interactive mode we'll
            # never get a label

        # otherwise, we use the last output the model generated
        if self_message is not None:
            last_reply = self_message['text_1']
            self.history.add_reply(last_reply)
            return

        raise RuntimeError("Unexpected case in self_observe.")
        
    def state_dict(self):
        """
        Get the model states for saving.

        Overriden to include longest_label
        """
        states = super().state_dict()
        if hasattr(self.model, 'module'):
            states['longest_label'] = self.model.module.longest_label
        else:
            states['longest_label'] = self.model.longest_label

        return states

    def load(self, path):
        """
        Return opt and model states.
        """
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        # set loaded states if applicable
        self.model.load_state_dict(states['model'])
        if 'longest_label' in states:
            self.model.longest_label = states['longest_label']
        return states

    def is_valid(self, obs):
        contains_empties = []
        for i in range(2):
            #print(i, obs[f'text_vec_{i}'], obs[f'text_vec_{i}'].shape[0])
            contains_empties.append(obs[f'text_vec_{i}'].shape[0] == 0)
        if self.is_training and any(contains_empties):
            warn_once(
                'seq2seq got an empty input sequence (text_vec) during training. '
                'Skipping this example, but you should check your dataset and '
                'preprocessing.'
            )
        elif not self.is_training and contains_empties:
            warn_once(
                'seq2seq got an empty input sequence (text_vec) in an '
                'evaluation example! This may affect your metrics!'
            )
        return [not x for x in contains_empties]
       
    def build_criterion(self):
        # set up criteria
        if self.opt.get('numsoftmax', 1) > 1:
            return nn.NLLLoss(ignore_index=self.NULL_IDX, reduction='none')
        else:
            return nn.CrossEntropyLoss(ignore_index=self.NULL_IDX, reduction='none') 

    def compute_loss(self, batch, return_output=False):
        """
        Compute and return the loss for the given batch.

        Easily overridable for customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch['label_1_vec'] is None or batch['label_2_vec'] is None:
            raise ValueError('Cannot compute loss without a label.')

        #TODO HRED: change model forward function to implement HRED 
        # model output: (output_1, output_2)
        model_output = self.model(*self._model_input(batch), ys1=batch.label_1_vec, ys2 = batch.label_2_vec)
        scores_1, preds_1, scores_2, preds_2,  *_ = model_output
        score_1_view = scores_1.view(-1, scores_1.size(-1))
        loss_1 = self.criterion(score_1_view, batch.label_1_vec.view(-1))
        loss_1 = loss_1.view(scores_1.shape[:-1]).sum(dim=1)

        score_2_view = scores_2.view(-1, scores_2.size(-1))
        loss_2 = self.criterion(score_2_view, batch.label_2_vec.view(-1))
        loss_2 = loss_2.view(scores_2.shape[:-1]).sum(dim=1)

        # save loss to metrics
        notnull_1 = batch.label_1_vec.ne(self.NULL_IDX)
        target_tokens_1 = notnull_1.long().sum(dim=-1)
        #target_tokens_1 = notnull_1.long().sum().item()
        correct_1 = ((batch.label_1_vec == preds_1) * notnull_1).sum(dim=-1)
        notnull_2 = batch.label_2_vec.ne(self.NULL_IDX)
        #target_tokens_2 = notnull_2.long().sum().item()
        target_tokens_2 = notnull_2.long().sum(dim=-1)
        correct_2 = ((batch.label_2_vec == preds_2) * notnull_2).sum(dim=-1)
        
        target_tokens = torch.cat((target_tokens_1, target_tokens_2))
        correct = torch.cat((correct_1, correct_2)) 
        total_losses = torch.cat((loss_1, loss_2))

        num_tokens = target_tokens.sum().item()
        

        self.record_local_metric('loss_1', AverageMetric.many(loss_1, target_tokens_1))
        self.record_local_metric('ppl_1', PPLMetric.many(loss_1, target_tokens_1))
        self.record_local_metric(
                    'token_acc_1', AverageMetric.many(correct_1, target_tokens_1)
                                )

        self.record_local_metric('loss_2', AverageMetric.many(loss_2, target_tokens_2))
        self.record_local_metric('ppl_2', PPLMetric.many(loss_2, target_tokens_2))
        self.record_local_metric(
                    'token_acc_2', AverageMetric.many(correct_2, target_tokens_2)
                                )
        loss = total_losses.sum()
        loss /= num_tokens # average loss per token

        if return_output:
            return (loss, model_output)
        else:
            return loss

    def eval_step(self, batch):
        """
        Evaluate a single batch of examples.
        """
        if batch.text_0_vec is None and batch.image is None:
            return
        if batch.text_0_vec is not None:
            bsz = batch.text_0_vec.size(0)
        else:
            bsz = len(batch.image)
        self.model.eval()
        cand_scores = None
        token_losses = None

        if "labels_1_vec" in batch and batch.labels_1_vec is not None:
            # calculate loss on targets with teacher forcing
            loss, model_output = self.compute_loss(batch, return_output=True)
            if self.output_token_losses:
                token_losses = self._construct_token_losses(
                    batch.labels_1_vec_1, model_output
                )

        preds = None
        if self.skip_generation:
            warn_once(
                "--skip-generation does not produce accurate metrics beyond ppl",
                RuntimeWarning,
            )
        else:
            maxlen = self.label_truncate or 256
            

            beam_preds_scores_1, _ = self._generate(batch, self.beam_size, maxlen, index=1)
            preds_1, scores_1 = zip(*beam_preds_scores_1)

        #cand_choices_0 = None
        cand_choices_1 = None

        # TODO: abstract out the scoring here
        if self.rank_candidates:
            # compute roughly ppl to rank candidates
            cand_choices = []
            encoder_states = self.model.encoder(*self._encoder_input(batch, index))
            for i in range(bsz):
                num_cands = len(batch.candidate_vecs[i])
                enc = self.model.reorder_encoder_states(encoder_states, [i] * num_cands)
                cands, _ = self._pad_tensor(batch.candidate_vecs[i])
                scores, _ = self.model.decode_forced(enc, cands)
                cand_losses = F.cross_entropy(
                    scores.view(num_cands * cands.size(1), -1),
                    cands.view(-1),
                    reduction='none',
                ).view(num_cands, cands.size(1))
                # now cand_losses is cands x seqlen size, but we still need to
                # check padding and such
                mask = (cands != self.NULL_IDX).float()
                cand_scores = (cand_losses * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
                _, ordering = cand_scores.sort()
                cand_choices.append([batch.candidates[i][o] for o in ordering])

        #text_0 = [self._v2t(p) for p in preds_0] if preds_0 is not None else None
        if len(self.history.history_strings) > 0:
            #print(f"setting text 0 to {self.history.history_strings[-1]}") 
            text_0 = self.history.history_strings[-1]
        else:
            text_0  = "__SILENCE__"
        #print(f'preds 1 is {preds_1}') 
        text_1 = [self._v2t(p) for p in preds_1] if preds_1 is not None else None
        if text_1 == '':
            text_1 = "__SILENCE__"
        #if text and self.compute_tokenized_bleu:
        #    # compute additional bleu scores
        #    self._compute_fairseq_bleu(batch, preds)
        #    self._compute_nltk_bleu(batch, text)
        return HREDOutput(text_0 = text_0, text_1 = text_1, text_0_candidates = None , text_1_candidates = cand_choices_1, token_losses=token_losses)

    def _generate(self, batch, beam_size, max_ts, index = 0, context_vector=None):
        """
        Generate an output with beam search.

        Depending on the options, this may perform greedy/topk/nucleus generation.

        :param Batch batch:
            Batch structure with input and labels
        :param int beam_size:
            Size of each beam during the search
        :param int max_ts:
            the maximum length of the decoded sequence

        :return:
            tuple (beam_pred_scores, n_best_pred_scores, beams)

            - beam_preds_scores: list of (prediction, score) pairs for each sample in
              Batch
            - n_best_preds_scores: list of n_best list of tuples (prediction, score)
              for each sample from Batch
            - beams :list of Beam instances defined in Beam class, can be used for any
              following postprocessing, e.g. dot logging.
        """
        model = self.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        encoder_states = model.encoder(*self._encoder_input(batch, index))
        if index == 0:
            if batch.text_0_vec is not None:
                dev = batch.text_0_vec.device

            bsz = (
                len(batch.text_0_lengths)
                if batch.text_0_lengths is not None
                else len(batch.image)
            )
            if batch.text_0_vec is not None:
                batchsize = batch.text_0_vec.size(0)
                beams = [
                    self._treesearch_factory(dev).set_context(
                        self._get_context(batch, batch_idx, index)
                    )
                    for batch_idx in range(batchsize)
                ]
            else:
                beams = [self._treesearch_factory(dev) for _ in range(bsz)]

        if index == 1:
            if batch.text_1_vec is not None:
                dev = batch.text_1_vec.device

            bsz = (
                len(batch.text_1_lengths)
                if batch.text_1_lengths is not None
                else len(batch.image)
            )
            if batch.text_1_vec is not None:
                batchsize = batch.text_1_vec.size(0)
                beams = [
                    self._treesearch_factory(dev).set_context(
                        self._get_context(batch, batch_idx, index)
                    )
                    for batch_idx in range(batchsize)
                ]
            else:
                beams = [self._treesearch_factory(dev) for _ in range(bsz)]

        # repeat encoder outputs and decoder inputs
        decoder_input = (
            torch.LongTensor([self.START_IDX]).expand(bsz * beam_size, 1).to(dev)
        )

        inds = torch.arange(bsz).to(dev).unsqueeze(1).repeat(1, beam_size).view(-1)
        encoder_states = model.reorder_encoder_states(encoder_states, inds)
        incr_state = None

        if context_vector is None:
            context_vector = torch.zeros((bsz, model.csz))

        for _ts in range(max_ts):
            if all((b.is_done() for b in beams)):
                # exit early if possible
                break

            score, incr_state = model.decoder(decoder_input[:,:20], encoder_states, context_vector.cuda())
            #print(decoder_input[:,:20])
            # only need the final hidden state to make the word prediction
            score = score[:, -1:, :]
            score = model.output(score)
            # score contains softmax scores for bsz * beam_size samples
            score = score.view(bsz, beam_size, -1)
            score = F.log_softmax(score, dim=-1)
            for i, b in enumerate(beams):
                if not b.is_done():
                    b.advance(score[i])
            incr_state_inds = torch.cat(
                [
                    beam_size * i + b.get_backtrack_from_current_step()
                    for i, b in enumerate(beams)
                ]
            )
            incr_state = model.reorder_decoder_incremental_state(
                incr_state, incr_state_inds
            )
            decoder_input = torch.index_select(decoder_input, 0, incr_state_inds)
            selection = torch.cat(
                [b.get_output_from_current_step() for b in beams]
            ).unsqueeze(-1)
            decoder_input = torch.cat([decoder_input, selection], dim=-1)

        # get all finilized candidates for each sample (and validate them)
        n_best_beam_preds_scores = [b.get_rescored_finished() for b in beams]

        # get the top prediction for each beam (i.e. minibatch sample)
        beam_preds_scores = [n_best_list[0] for n_best_list in n_best_beam_preds_scores]

        return beam_preds_scores, beams

    def _get_context(self, batch, batch_idx, index):
        """
        Set the beam context for n-gram context blocking.

        Intentionally overridable for more complex model histories.
        """
        if index == 0:
            return batch.text_0_vec[batch_idx]
        elif index == 1:
            return batch.text_1_vec[batch_idx]
        else:
            raise IndexError("This is not a valid index") 

    def _encoder_input(self, batch, index):
        if index == 0:
            return (batch.text_0_vec,)
        elif index == 1:
            return (batch.text_1_vec,)
        else:
            raise IndexError("This is not a valid index") 
    def _model_input(self, batch):
        return (batch.text_0_vec, batch.text_1_vec)
