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


  return


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
    data_format: `str`. Only works for "channels_last" currently. If use_5d_mode
      is True, its shape is `[batch, time, height, width, channels]`. Otherwise
      `[batch*time, height, width, channels]`.
  Returns:
    The output `Tensor` after concatenation.
  """
  data_format = tf.keras.backend.image_data_format()
  assert data_format == 'channels_last'


  inputs =

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
    #todo: fill here

    return

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

    #todo: fill here

    return



def fusion_with_peer_attention(inputs: List[tf.Tensor],
                               index: int = None,
                               attention_mode=None,
                               attention_in=None,
                               use_5d_mode: bool = False,
                               model_edge_weights: Optional[List[Any]] = None):
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
    #todo: fill here

  per_channel_inps = _ApplyEdgeWeight(
    weights_shape=[len(inputs)],
    index=index,
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




  return tf.add_n(inps)



def object_conv_stem(inputs):
  """Layers for an object input stem.
  It expects its input tensor to have a separate channel for each object class.
  Args:
    inputs: A `Tensor`.
  Returns:
    The output `Tensor`.
  """
  inputs = tf.keras.layers.MaxPooling2D(
    inputs=inputs,
    pool_size=4,
    strides=4,
    padding='SAME',
  )
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
               max_pool_preditions: bool = False,
               **kwargs):
    inputs =
    outputs =

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
    #todo: fill here
    **kwargs
  )
  return AssembleNetPlusModel(
    backbone,
    #todo: fill here
    **kwargs)

@backbone_factory.register_backbone_builder('assemblenet_plus')
def build_assemblenet_plus(
    input_specs: tf.keras.layers.InputSpec,
    model_config: cfg.Backbone3D,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:
  """Builds assemblenet++ backbone."""
  del l2_regularizer


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
      #todo: fill here
      )
  return model

