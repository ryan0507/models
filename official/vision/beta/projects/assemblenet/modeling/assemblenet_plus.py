# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains definitions for the AssembleNet++ [2] models (without object input).
Requires the AssembleNet++ architecture to be specified in
FLAGS.model_structure (and optionally FLAGS.model_edge_weights). This is
identical to the form described in assemblenet.py for the AssembleNet. Please
check assemblenet.py for the detailed format of the model strings.
AssembleNet++ adds `peer-attention' to the basic AssembleNet, which allows each
conv. block connection to be conditioned differently based on another block [2].
It is a form of channel-wise attention. Note that we learn to apply attention
independently for each frame.
The `peer-attention' implementation in this file is the version that enables
one-shot differentiable search of attention connectivity (Fig. 2 in [2]), using
a softmax weighted summation of possible attention vectors.
[2] Michael S. Ryoo, AJ Piergiovanni, Juhana Kangaspunta, Anelia Angelova,
    AssembleNet++: Assembling Modality Representations via Attention
    Connections. ECCV 2020
    https://arxiv.org/abs/2008.08072
In order to take advantage of object inputs, one will need to set the flag
FLAGS.use_object_input as True, and provide the list of input tensors as an
input to the network, as shown in run_asn_with_object.py. This will require a
pre-processed object data stream.
It uses (2+1)D convolutions for video representations. The main AssembleNet++
takes a 4-D (N*T)HWC tensor as an input (i.e., the batch dim and time dim are
mixed), and it reshapes a tensor to NT(H*W)C whenever a 1-D temporal conv. is
necessary. This is to run this on TPU efficiently.
"""

import functools
import math
from typing import Any, Mapping, List, Callable, Optional

from absl import logging
import numpy as np
import tensorflow as tf

from official.vision.beta.modeling import factory_3d as model_factory
from official.vision.beta.modeling.backbones import factory as backbone_factory
from official.vision.beta.projects.assemblenet.configs import assemblenet as cfg
from official.vision.beta.projects.assemblenet.modeling import rep_flow_2d_layer as rf
from official.vision.beta.projects.assemblenet.modeling import assemblenet as asn

layers = tf.keras.layers

def softmax_merge_peer_attentions(peers):
  """Merge multiple peer-attention vectors with softmax weighted sum.
  Summation weights are to be learned.
  Args:
    peers: A list of `Tensors` of size `[batch*time, channels]`.
  Returns:
    The output `Tensor` of size `[batch*time, channels].
  """
  data_format = tf.keras.backend.image_data_format()
  assert data_format == 'channels_last'

  initial_attn_weights = tf.keras.initializers.TruncatedNormal(stddev=0.01)([len(peers)])
  attn_weights = tf.keras.layers.Softmax()(input = initial_attn_weights)

  weighted_peers = []
  for i,peer in enumerate(peers):
    weighted_peers.append(attn_weights[i]*peer)

  return tf.add_n(weighted_peers)


def apply_attention(inputs,
                    attention_mode=None,
                    attention_in=None,
                    use_5d_mode=False):
  """Applies peer-attention or self-attention to the input tensor.

  Depending on the attention_mode, this function either applies channel-wise
  self-attention or peer-attention. For the peer-attention, the function
  combines multiple candidate attention vectors (given as attention_in), by
  learning softmax-sum weights described in the AssembleNet++ paper. Note that
  the attention is applied individually for each frame, which showed better
  accuracies than using video-level attention.

  Args:
    inputs: A `Tensor`. Either 4D or 5D, depending of use_5d_mode.
    attention_mode: `str` specifying mode. If not `peer', does self-attention.
    attention_in: A list of `Tensors' of size [batch*time, channels].
    use_5d_mode: `bool` indicating whether the inputs are in 5D tensor or 4D.

  Returns:
    The output `Tensor` after concatenation.
  """
  data_format = tf.keras.backend.image_data_format()
  assert data_format == 'channels_last'

  if use_5d_mode:
    h_channel_loc = 2
  else:
    h_channel_loc = 1

  if attention_mode == 'peer':
    attn = softmax_merge_peer_attentions(attention_in)
  else:
    attn = tf.math.reduce_mean(inputs, [h_channel_loc, h_channel_loc+1])
  attn = tf.keras.layers.Dense(
    units=inputs.shape[-1],
    kernel_initializer=tf.random_normal_initializer(stddev=.01))(
    inputs=attn)
  attn = tf.math.sigmoid(attn)
  channel_attn = tf.expand_dims(tf.expand_dims(attn, h_channel_loc), h_channel_loc)

  inputs = tf.math.multiply(inputs, channel_attn)

  return inputs


class _ApplyEdgeWeight(layers.Layer):
  """Multiply weight on each input tensor.

  A weight is assigned for each connection (i.e., each input tensor). This layer
  is used by the fusion_with_peer_attention to compute the weighted inputs.
  """

  def __init__(self,
               weights_shape,
               index: int = None,
               attention_mode: str = None,
               attention_in: tf.Tensor = None,
               use_5d_mode: bool = False,
               model_edge_weights: Optional[List[Any]] = None,
               num_object_classes: int = None, #todo: newly added - check https://github.com/google-research/google-research/blob/bc7791a7770ce3466fe8df84bec65fed0b77ecb8/assemblenet/run_asn_with_object.py#L57
               **kwargs):
    """Constructor.

    Args:
      inputs: A list of `Tensors`. Either 4D or 5D, depending of use_5d_mode.
      index: `int` index of the block within the AssembleNet architecture. Used
        for summation weight initial loading.
      attention_mode: `str` specifying mode. If not `peer', does self-attention.
      attention_in: A list of `Tensors' of size [batch*time, channels].
      use_5d_mode: `bool` indicating whether the inputs are in 5D tensor or 4D.
      model_edge_weights: AssembleNet model structure connection weights in the
        string format.
      **kwargs: pass through arguments.

    Returns:
      The output `Tensor` after concatenation.
    """

    super(_ApplyEdgeWeight, self).__init__(**kwargs)

    self._weights_shape = weights_shape
    self._index = index
    self._attention_mode = attention_mode
    self._attention_in = attention_in
    self._use_5d_mode = use_5d_mode
    self._model_edge_weights = model_edge_weights
    self._num_object_classes = num_object_classes
    data_format = tf.keras.backend.image_data_format()
    assert data_format == 'channels_last'

  def get_config(self):
    config = {
      'weights_shape': self._weights_shape,
      'index': self._index,
      'attention_mode': self._attention_mode,
      'attention_in' : self._attention_in,
      'use_5d_mode': self._use_5d_mode,
      'model_edge_weights': self._model_edge_weights,
    }
    base_config = super(_ApplyEdgeWeight, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape: tf.TensorShape):
    if self._weights_shape[0] == 1:
      self._edge_weights = 1.0
      return

    if self._index is None or not self._model_edge_weights:
      self._edge_weights = self.add_weight(
        shape=self._weights_shape,
        initializer=tf.keras.initializers.TruncatedNormal(
          mean=0.0, stddev=0.01),
        trainable=True,
        name='agg_weights')
    else:
      initial_weights_after_sigmoid = np.asarray(
        self._model_edge_weights[self._index][0]).astype('float32')
      # Initial_weights_after_sigmoid is never 0, as the initial weights are
      # based the results of a successful connectivity search.
      initial_weights = -np.log(1. / initial_weights_after_sigmoid - 1.)
      self._edge_weights = self.add_weight(
        shape=self._weights_shape,
        initializer=tf.constant_initializer(initial_weights),
        trainable=False,
        name='agg_weights')

  def call(self,
           inputs: List[tf.Tensor],
           training: bool = None) -> Mapping[Any, List[tf.Tensor]]:
    use_5d_mode = self._use_5d_mode
    dtype = inputs[0].dtype
    assert len(inputs) > 1

    if use_5d_mode:
      h_channel_loc = 2
    else:
      h_channel_loc = 1

    # get smallest spatial size and largest channels
    sm_size = [10000, 10000]
    lg_channel = 0
    for inp in inputs:
      # assume batch X height x width x channels
      sm_size[0] = min(sm_size[0], inp.shape[h_channel_loc])
      sm_size[1] = min(sm_size[1], inp.shape[h_channel_loc + 1])
      # Note that, when using object inputs, object channel sizes are usually big.
      # Since we do not want the object channel size to increase the number of
      # parameters for every fusion, we exclude it when computing lg_channel.
      if inp.shape[-1] > lg_channel and inp.shape[-1] != self._num_object_classes:  # pylint: disable=line-too-long
        lg_channel = inp.shape[3]

    # loads or creates weight variables to fuse multiple inputs
    weights = tf.math.sigmoid(tf.cast(self._edge_weights, dtype))

    # Compute weighted inputs. We group inputs with the same channels.
    per_channel_inps = dict({0: []})
    for i, inp in enumerate(inputs):
      if inp.shape[h_channel_loc] != sm_size[0] or inp.shape[h_channel_loc + 1] != sm_size[1]:  # pylint: disable=line-too-long
        assert sm_size[0] != 0
        ratio = (inp.shape[h_channel_loc] + 1) // sm_size[0]
        if use_5d_mode:
          inp = tf.keras.layers.MaxPool3D([1, ratio, ratio], [1, ratio, ratio],
                                          padding='same')(
                                              inp)
        else:
          inp = tf.keras.layers.MaxPool2D([ratio, ratio], ratio,
                                          padding='same')(
                                              inp)

      weights = tf.cast(weights, inp.dtype)
      if inp.shape[-1] in per_channel_inps:
        per_channel_inps[inp.shape[-1]].append(weights[i] * inp)
      else:
        per_channel_inps.update({inp.shape[-1]: [weights[i] * inp]})

    # Implementation of connectivity with peer-attention
    if self._attention_mode:
      for key, channel_inps in per_channel_inps.items():
        for idx in range(len(channel_inps)):
          with tf.name_scope('Connection_' + str(key) + '_' + str(idx)):
            channel_inps[idx] = apply_attention(channel_inps[idx],
                                                self._attention_mode,
                                                self._attention_in,
                                                self._use_5d_mode)

    return per_channel_inps


def fusion_with_peer_attention(inputs: List[tf.Tensor],
                               index: int = None,
                               attention_mode=None,
                               attention_in=None,
                               use_5d_mode: bool = False,
                               model_edge_weights: Optional[List[Any]] = None,
                               num_object_classes: int = None): # todo: newly added - check https://github.com/google-research/google-research/blob/bc7791a7770ce3466fe8df84bec65fed0b77ecb8/assemblenet/run_asn_with_object.py#L57
  """Weighted summation of multiple tensors, while using peer-attention.

  Summation weights are to be learned. Uses spatial max pooling and 1x1 conv.
  to match their sizes. Before the summation, each connection (i.e., each input)
  itself is scaled with channel-wise peer-attention. Notice that attention is
  applied for each connection, conditioned based on attention_in.

  Args:
    inputs: A list of `Tensors`. Either 4D or 5D, depending of use_5d_mode.
    index: `int` index of the block within the AssembleNet architecture. Used
      for summation weight initial loading.
    attention_mode: `str` specifying mode. If not `peer', does self-attention.
    attention_in: A list of `Tensors' of size [batch*time, channels].
    use_5d_mode: `bool` indicating whether the inputs are in 5D tensor or 4D.
    model_edge_weights: AssembleNet model structure connection weights in the
  string format.

  Returns:
    The output `Tensor` after concatenation.
  """
  if use_5d_mode:
    h_channel_loc = 2
    conv_function = asn.conv3d_same_padding
  else:
    h_channel_loc = 1
    conv_function = asn.conv2d_fixed_padding

  # If only 1 input.
  if len(inputs) == 1:
    inputs[0] = apply_attention(inputs[0],
                                attention_mode,
                                attention_in,
                                use_5d_mode)
    return inputs[0]

  # get smallest spatial size and largest channels
  sm_size = [10000, 10000]
  lg_channel = 0
  for inp in inputs:
    # assume batch X height x width x channels
    sm_size[0] = min(sm_size[0], inp.shape[h_channel_loc])
    sm_size[1] = min(sm_size[1], inp.shape[h_channel_loc + 1])
    # Note that, when using object inputs, object channel sizes are usually big.
    # Since we do not want the object channel size to increase the number of
    # parameters for every fusion, we exclude it when computing lg_channel.
    if inp.shape[-1] > lg_channel and inp.shape[-1] != num_object_classes:  # pylint: disable=line-too-long
      lg_channel = inp.shape[3]

  per_channel_inps = _ApplyEdgeWeight(
    weights_shape=[len(inputs)],
    index=index,
    attention_mode=attention_mode,
    attention_in=attention_in,
    use_5d_mode=use_5d_mode,
    model_edge_weights=model_edge_weights)(
    inputs)

  # Adding 1x1 conv layers (to match channel size) and fusing all inputs.
  # We add inputs with the same channels first before applying 1x1 conv to save
  # memory.
  inps = []
  for key, channel_inps in per_channel_inps.items():
    if len(channel_inps) < 1:
      continue
    if len(channel_inps) == 1:
      if key == lg_channel:
        inp = channel_inps[0]
      else:
        inp = conv_function(
          channel_inps[0], lg_channel, kernel_size=1, strides=1)
      inps.append(inp)
    else:
      if key == lg_channel:
        inp = tf.add_n(channel_inps)
      else:
        inp = conv_function(
          channel_inps[0], lg_channel, kernel_size=1, strides=1)
      inps.append(inp)

  return tf.add_n(inps)


def object_conv_stem(inputs):
  """Layers for an object input stem.
  It expects its input tensor to have a separate channel for each object class.
  Args:
    inputs: A `Tensor`.
  Returns:
    The output `Tensor`.
  """
  inputs = tf.keras.layers.MaxPool2D(
    pool_size=4, strides=4, padding='SAME')(
        inputs=inputs)
  inputs = tf.identity(inputs, 'initial_max_pool')

  return inputs


class AssembleNetPlus(tf.keras.Model):
  """AssembleNet++ backbone."""

  def __init__(
      self,
      block_fn,
      num_blocks: List[int],
      num_frames: int,
      model_structure: List[Any],
      input_specs: layers.InputSpec = layers.InputSpec(
        shape=[None, None, None, None, 3]),
      model_edge_weights: Optional[List[Any]] = None,
      bn_decay: float = rf.BATCH_NORM_DECAY,
      bn_epsilon: float = rf.BATCH_NORM_EPSILON,
      use_sync_bn: bool = False,
      use_object_input: bool = False, #todo: newly added - doc later
      attention_mode: str = None, #todo: newly added - doc later
      **kwargs):
    """Generator for AssembleNet++ models.
    
    Args:
      block_fn: `function` for the block to use within the model. Currently only
        has `bottleneck_block_interleave as its option`.
      layers: list of 4 `int`s denoting the number of blocks to include in each of
        the 4 block groups. Each group consists of blocks that take inputs of the
        same resolution.
      num_classes: `int` number of possible classes for video classification.
      data_format: `str` either "channels_first" for `[batch*time, channels,
        height, width]` or "channels_last" for `[batch*time, height, width,
        channels]`.

    Returns:
      Model `function` that takes in `inputs` and `is_training` and returns the
      output `Tensor` of the AssembleNet model.
    """
    inputs = tf.keras.Input(shape=input_specs.shape[1:])
    data_format = tf.keras.backend.image_data_format()

    # Creation of the model graph.
    logging.info('model_structure=%r', model_structure)
    logging.info('model_structure=%r', model_structure)
    logging.info('model_edge_weights=%r', model_edge_weights)
    structure = model_structure

    if use_object_input:
      original_inputs = inputs[0]
      object_inputs = inputs[1]
    else:
      original_inputs = inputs
      object_inputs = None

    original_num_frames = num_frames
    assert num_frames > 0, f'Invalid num_frames {num_frames}'

    grouping = {-3: [], -2: [], -1: [], 0: [], 1: [], 2: [], 3: []}
    for i in range(len(structure)):
      grouping[structure[i][0]].append(i)

    stem_count = len(grouping[-3]) + len(grouping[-2]) + len(grouping[-1])

    assert stem_count != 0
    stem_filters = 128 // stem_count

    if len(input_specs.shape) == 5:
      first_dim = (
          input_specs.shape[0] * input_specs.shape[1]
          if input_specs.shape[0] and input_specs.shape[1] else -1)
      reshape_inputs = tf.reshape(inputs, (first_dim,) + input_specs.shape[2:])
    elif len(input_specs.shape) == 4:
      reshape_inputs = original_inputs
    else:
      raise ValueError(
          f'Expect input spec to be 4 or 5 dimensions {input_specs.shape}')

    if grouping[-2]:
      # Instead of loading optical flows as inputs from data pipeline, we are
      # applying the "Representation Flow" to RGB frames so that we can compute
      # the flow within TPU/GPU on fly. It's essentially optical flow since we
      # do it with RGBs.
      axis = 3 if data_format == 'channels_last' else 1
      flow_inputs = rf.RepresentationFlow(
          original_num_frames,
          depth=reshape_inputs.shape.as_list()[axis],
          num_iter=40,
          bottleneck=1)(
              reshape_inputs)
    streams = []

    for i in range(len(structure)):
      with tf.name_scope('Node_' + str(i)):
        if structure[i][0] == -1:
          inputs = asn.rgb_conv_stem(
              reshape_inputs,
              original_num_frames,
              stem_filters,
              temporal_dilation=structure[i][1],
              bn_decay=bn_decay,
              bn_epsilon=bn_epsilon,
              use_sync_bn=use_sync_bn)
          streams.append(inputs)
        elif structure[i][0] == -2:
          inputs = asn.flow_conv_stem(
              flow_inputs,
              stem_filters,
              temporal_dilation=structure[i][1],
              bn_decay=bn_decay,
              bn_epsilon=bn_epsilon,
              use_sync_bn=use_sync_bn)
          streams.append(inputs)
        elif structure[i][0] == -3:
          # In order to use the object inputs, you need to feed your object
          # input tensor here.
          inputs = object_conv_stem(object_inputs)
          streams.append(inputs)
        else:
          block_number = structure[i][0]

          combined_inputs = [streams[structure[i][1][j]]
                             for j in range(0, len(structure[i][1]))]

          logging.info(grouping)
          nodes_below = []
          for k in range(-3, structure[i][0]):
            nodes_below = nodes_below + grouping[k]

          peers = []
          if attention_mode:
            lg_channel = -1
            logging.info(nodes_below)
            for k in nodes_below:
              logging.info(streams[k].shape)
              lg_channel = max(streams[k].shape[3], lg_channel)

            for node_index in nodes_below:
              attn = tf.reduce_mean(streams[node_index], [1,2])

              attn = tf.keras.layers.Dense(
                units=lg_channel,
                kernel_initializer=tf.random_normal_initializer(stddev=.01))(
                  inputs=attn)
              peers.append(attn)

          combined_inputs = fusion_with_peer_attention(
            combined_inputs,
            index=i,
            attention_mode=attention_mode,
            attention_in=peers,
            use_5d_mode=False)

          graph = asn.block_group(
            inputs=combined_inputs,
            filters=structure[i][2],
            block_fn=block_fn,
            blocks=num_blocks[block_number],
            strides=structure[i][4],
            name='block_group' + str(i),
            block_level=structure[i][0],
            num_frames=num_frames,
            temporal_dilation=structure[i][3])

          streams.append(graph)

    super(AssembleNetPlus, self).__init__(
      inputs=original_inputs, outputs=streams, **kwargs)


@tf.keras.utils.register_keras_serializable(package='Vision')
class AssembleNetPlusModel(tf.keras.Model):
  """An AssembleNet++ model builder."""

  def __init__(self,
               backbone,
               num_classes,
               num_frames: int,
               model_structure: List[Any],
               input_specs: Mapping[str, tf.keras.layers.InputSpec] = None,
               max_pool_predictions: bool = False,
               **kwargs):
    if not input_specs:
      input_specs = {
          'image': layers.InputSpec(shape=[None, None, None, None, 3])
      }
    self._self_setattr_tracking = False
    self._config_dict = {
        'backbone': backbone,
        'num_classes': num_classes,
        'num_frames': num_frames,
        'input_specs': input_specs,
        'model_structure': model_structure,
    }
    self._input_specs = input_specs
    self._backbone = backbone
    grouping = {-3: [], -2: [], -1: [], 0: [], 1: [], 2: [], 3: []}
    for i in range(len(model_structure)):
      grouping[model_structure[i][0]].append(i)

    inputs = {
        k: tf.keras.Input(shape=v.shape[1:]) for k, v in input_specs.items()
    }
    streams = self._backbone(inputs['image'])

    outputs = asn.multi_stream_heads(
        streams,
        grouping[3],
        num_frames,
        num_classes,
        max_pool_preditions=max_pool_predictions)

    super(AssembleNetPlusModel, self).__init__(
        inputs=inputs, outputs=outputs, **kwargs)

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    return dict(backbone=self.backbone)

  @property
  def backbone(self):
    return self._backbone

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


ASSEMBLENET_SPECS = {
      26: {
          'block': asn.bottleneck_block_interleave,
          'layers': [2, 2, 2, 2]
      },
      38: {
          'block': asn.bottleneck_block_interleave,
          'layers': [2, 4, 4, 2]
      },
      50: {
          'block': asn.bottleneck_block_interleave,
          'layers': [3, 4, 6, 3]
      },
      68: {
          'block': asn.bottleneck_block_interleave,
          'layers': [3, 4, 12, 3]
      },
      77: {
          'block': asn.bottleneck_block_interleave,
          'layers': [3, 4, 15, 3]
      },
      101: {
          'block': asn.bottleneck_block_interleave,
          'layers': [3, 4, 23, 3]
      },
  }

def assemblenet_plus(assemblenet_depth: int,
                   num_classes: int,
                   num_frames: int,
                   model_structure: List[Any],
                   input_specs: layers.InputSpec = layers.InputSpec(
                       shape=[None, None, None, None, 3]),
                   model_edge_weights: Optional[List[Any]] = None,
                   bn_decay: float = rf.BATCH_NORM_DECAY,
                   bn_epsilon: float = rf.BATCH_NORM_EPSILON,
                   use_sync_bn: bool = False,
                   use_object_input: bool = False,  # todo: newly added - doc later
                   attention_mode: str = None,  # todo: newly added - doc later
                   max_pool_preditions: bool = False,
                     **kwargs):
  """Returns the AssembleNet++ model for a given size and number of output classes."""

  data_format = tf.keras.backend.image_data_format()
  assert data_format == 'channels_last'

  if assemblenet_depth not in ASSEMBLENET_SPECS:
    raise ValueError('Not a valid assemblenet_depth:', assemblenet_depth)

  input_specs_dict = {'image': input_specs}
  params = ASSEMBLENET_SPECS[assemblenet_depth]
  backbone = AssembleNetPlus(
    block_fn=params['block'],
    num_blocks=params['num_blocks'],
    num_frames=num_frames,
    model_structure=model_structure,
    input_specs=input_specs,
    model_edge_weights=model_edge_weights,
    bn_decay=bn_decay,
    bn_epsilon=bn_epsilon,
    use_sync_bn=use_sync_bn,
    use_object_input=use_object_input,
    attention_mode=attention_mode,
    **kwargs)
  return AssembleNetPlusModel(
    backbone,
    num_classes=num_classes,
    num_frames=num_frames,
    model_structure=model_structure,
    input_specs=input_specs_dict,
    max_pool_preditions=max_pool_preditions,
    **kwargs)


@backbone_factory.register_backbone_builder('assemblenet_plus')
def build_assemblenet_plus(
    input_specs: tf.keras.layers.InputSpec,
    model_config: cfg.Backbone3D,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:
  """Builds assemblenet++ backbone."""
  del l2_regularizer

  backbone_type = model_config.backbone.type
  backbone_cfg = model_config.backbone.get()
  norm_activation_config = model_config.norm_activation
  assert backbone_type == 'assemblenet++'

  assemblenet_depth = int(backbone_cfg.model_id)
  if assemblenet_depth not in ASSEMBLENET_SPECS:
    raise ValueError('Not a valid assemblenet_depth:', assemblenet_depth)
  model_structure, model_edge_weights = cfg.blocks_to_flat_lists(
      backbone_cfg.blocks)
  params = ASSEMBLENET_SPECS[assemblenet_depth]
  block_fn = functools.partial(
      params['block'],
      use_sync_bn=norm_activation_config.use_sync_bn,
      bn_decay=norm_activation_config.norm_momentum,
      bn_epsilon=norm_activation_config.norm_epsilon)
  backbone = AssembleNetPlus(
      block_fn=block_fn,
      num_blocks=params['num_blocks'],
      num_frames=backbone_cfg.num_frames,
      model_structure=model_structure,
      input_specs=input_specs,
      model_edge_weights=model_edge_weights,
      use_sync_bn=norm_activation_config.use_sync_bn,
      bn_decay=norm_activation_config.norm_momentum,
      bn_epsilon=norm_activation_config.norm_epsilon,
      use_object_input=backbone_cfg.use_object_input, #todo: get from backbone config
      attention_mode=backbone_cfg.attention_mode) #todo: get from backbone config
  logging.info('Number of parameters in AssembleNet++ backbone: %f M.',
               backbone.count_params() / 10.**6)
  return backbone

@model_factory.register_model_builder('assemblenet_plus')
def build_assemblenet_plus_model(
    input_specs: tf.keras.layers.InputSpec,
    model_config: cfg.AssembleNetModel,
    num_classes: int,
    l2_regularizer: tf.keras.regularizers.Regularizer = None):
  """Builds assemblenet++ model."""
  input_specs_dict = {'image': input_specs}
  backbone = build_assemblenet_plus(input_specs, model_config, l2_regularizer)
  backbone_cfg = model_config.backbone.get()
  model_structure, _ = cfg.blocks_to_flat_lists(backbone_cfg.blocks)
  model = AssembleNetPlusModel(
      backbone,
      num_classes=num_classes,
      num_frames=backbone_cfg.num_frames,
      model_structure=model_structure,
      input_specs=input_specs_dict,
      max_pool_preditions=model_config.max_pool_preditions)
  return model

