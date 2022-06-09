'''
DUNE CVN model

Inspired by https://github.com/titu1994/keras-squeeze-excite-network/blob/master/se_resnet.py

References:
    - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
    - [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
'''
__version__ = '1.0'
__author__ = 'Saul Alonso-Monsalve, Leigh Howard Whitehead'
__email__ = 'saul.alonso.monsalve@cern.ch, leigh.howard.whitehead@cern.ch'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Activation, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, Conv2D
from tensorflow.keras.layers import add, concatenate, multiply, Permute
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

def DUNECVNModel(width=1,
                 weight_decay=1e-4,
                 weights=None):
    return SEResNetB(depth=[3, 4, 6, 3],
                    width=width,
                    weight_decay=weight_decay,
                    weights=weights)

def SEResNetB(initial_conv_filters=64,
              depth=[3, 4, 6, 3],
              filters=[64, 128, 256, 512],
              width=1,
              weight_decay=1e-4,
              weights=None,
              input_names=['view0','view1','view2'],
              input_shapes=[[500,500,1],[500,500,1],[500,500,1]],
              output_names=['is_antineutrino','flavour','interaction',\
                            'protons','pions','pizeros','neutrons'],
              output_neurons=[1,4,4,4,4,4,4]):
    ''' Instantiate the Squeeze and Excite ResNet architecture with branches.
    
    Args:
        initial_conv_filters: number of features for the initial convolution.
        depth: number or layers in the each block, defined as a list.
        filter: number of filters per block, defined as a list.
        width: width multiplier for the network (for Wide ResNets).
        weight_decay: weight decay (l2 norm).
        weights: path of HDF5 file with model weights.
        input_names: name of each input, defined as a list.
        input_shapes: shape of each input, defined as a list.
        output_names: name of each output, defined as a list.
        output_neurons: number of neurons of each output, defined as a list.

    Returns: a tf.keras model instance.
    '''
    assert len(depth) == len(filters), 'The length of filter increment list must match the length ' \
                                       'of the depth list.'
    assert len(input_names) == len(input_shapes), 'The length of input_names must match the length ' \
                                                  'of input_shapes.'
    assert len(output_names) == len(output_neurons), 'The length of output_names must match the length ' \
                                                     'of output_neurons.'
    # inputs
    inputs = [None]*len(input_names)
    for i in range(len(inputs)):
        inputs[i] = Input(shape=input_shapes[i], name=input_names[i])
    # generate architecture
    x = _create_se_resnet_with_branches(inputs, initial_conv_filters,
                          filters, depth, width, weight_decay)
    # outputs
    outputs = [None]*len(output_names)
    for i in range(len(outputs)):
        activation='sigmoid' if output_neurons[i]==1 else 'softmax'
        outputs[i] = Dense(output_neurons[i], use_bias=False, kernel_regularizer=l2(weight_decay),
                           activation=activation, name=output_names[i])(x)
    # create model
    model = Model(inputs=inputs, outputs=outputs, name='dunecvn')
    # load weights
    if weights:
        model.load_weights(weights, by_name=True)

    return model

def _resnet_block(input, filters, k=1, strides=(1, 1)):
    ''' Adds a pre-activation resnet block without bottleneck layers.

    Args:
        input: input tensor.
        filters: number of output filters.
        k: width factor.
        strides: strides of the convolution layer.

    Returns: a tf tensor.
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=channel_axis)(input)
    x = Activation('relu')(x)
    if strides != (1, 1) or init.shape[channel_axis] != filters * k:
        init = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)
    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)
    # squeeze and excite block
    x = squeeze_excite_block(x)
    m = add([x, init])

    return m

def squeeze_excite_block(input, ratio=16):
    ''' Create a squeeze-excite block.

    Args:
        input: input tensor.
        k: width factor.

    Returns: a tf tensor.
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    # se block
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)
    x = multiply([init, se])

    return x

def _create_se_resnet_with_branches(img_input, initial_conv_filters, filters,
                                    depth, width,  weight_decay):
    '''Creates the SE-ResNet architecture with specified parameters.

    Args:
        initial_conv_filters: number of features for the initial convolution.
        filters: number of filters per block, defined as a list.
        depth: number or layers in the each block, defined as a list.
        width: width multiplier for network (for Wide ResNet).
        weight_decay: weight_decay (l2 norm).

    Returns: a tf.keras Model.
    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    N = list(depth)

    # branches
    branches = []
    for i in range(len(img_input)):
        # block 1 (initial conv block)
        branch = Conv2D(initial_conv_filters, (7, 7), padding='same', use_bias=False, strides=(2, 2),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input[i])
        branch = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch)
        # block 2 (projection block)
        for i in range(N[0]):
            branch = _resnet_block(branch, filters[0], width)
        branches.append(branch)

    # concatenate branches
    x = concatenate(branches)

    # block 3 - N
    for k in range(1, len(N)):
        x = _resnet_block(x, filters[k], width)
        for i in range(N[k] - 1):
            x = _resnet_block(x, filters[k], width)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    return x

class CustomTrainStep(tf.keras.Model):
    def __init__(self, n_gradients=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in self.trainable_variables]

    def train_step(self, data):
        self.n_acum_step.assign_add(1)

        x, y = data
        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Calculate batch gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])

        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))

    def get_config(self):
        config = super().get_config()
        config['n_gradients'] = self.n_gradients
        return config
