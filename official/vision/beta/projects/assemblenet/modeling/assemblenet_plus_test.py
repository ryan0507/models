# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for assemblenet++ network."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.vision.beta.projects.assemblenet.modeling import assemblenet_plus as asnp
from official.vision.beta.projects.assemblenet.configs import assemblenet as asn_config

class AssembleNetPlusTest(parameterized.TestCase, tf.test.TestCase):
  @parameterized.parameters(
    (26),
    (38),
    (50),
    (68),
    (77),
    (101),
  )
  def test_network_creation(self, depth, ):

    batch_size = 2
    input_size = 224
    input_specs = tf.keras.layers.InputSpec(shape=[None, 32, input_size, input_size, 3])
    inputs = np.random.rand(batch_size, 32, input_size, input_size, 3)

    num_frames = 32
    model_structure = asn_config.full_asnp50_structure
    num_classes = 101 #ufc-101

    model = asnp.assemblenet_plus(assemblenet_depth=depth,
                                  num_classes=num_classes,
                                 num_frames=num_frames,
                                 model_structure=model_structure,
                                 input_specs=input_specs,)
                                 # use_object_input: bool = False,
                                 # attention_mode: str = None,
                                 # max_pool_preditions: bool = False,)


    outputs = model(inputs)
    self.assertAllEqual(outputs.shape.as_list(), [batch_size, num_classes])


if __name__ == '__main__':
  tf.test.main()