# Copyright 2018, Wenjia Bai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from ukbb_cardiac.common.network import *


def UNet(images, n_class, n_level, n_filter, n_block, training):
    """
        U-Net for segmenting an input image into n_class classes.

        images: NXYC
        """
    with tf.variable_scope('UNet'):
        net = {}
        x = images

        # Downsampling path
        # Learn fine-to-coarse features at each resolution level
        for l in range(0, n_level):
            with tf.variable_scope('conv{}'.format(l)):
                # If this is the first level (l = 0), keep the resolution.
                # Otherwise, convolve with a stride of 2, i.e. downsample by a
                # factor of 2.
                strides = 1 if l == 0 else 2
                # For each resolution level, perform n_block[l] times convolutions
                x = conv2d_bn_relu(x, filters=n_filter[l], training=training, kernel_size=3, strides=strides)
                for i in range(1, n_block[l]):
                    x = conv2d_bn_relu(x, filters=n_filter[l], training=training, kernel_size=3)
                net['conv{}'.format(l)] = x

        # Upsampling path
        l = n_level - 1
        with tf.variable_scope('conv{}_up'.format(l)):
            net['conv{}_up'.format(l)] = net['conv{}'.format(l)]

        for l in range(n_level - 2, -1, -1):
            with tf.variable_scope('conv{}_up'.format(l)):
                x = conv2d_transpose_bn_relu(net['conv{}_up'.format(l + 1)], filters=n_filter[l],
                                             training=training, kernel_size=3, strides=2)
                x = tf.concat([net['conv{}'.format(l)], x], axis=-1)
                for i in range(0, n_block[l]):
                    x = conv2d_bn_relu(x, filters=n_filter[l], training=training, kernel_size=3)
                net['conv{}_up'.format(l)] = x

        # Perform prediction
        with tf.variable_scope('conv_out'):
            # We only calculate logits, instead of softmax here because the loss
            # function tf.nn.softmax_cross_entropy() accepts the unscaled logits
            # and performs softmax internally for efficiency and numerical
            # stability reasons.
            # Refer to https://github.com/tensorflow/tensorflow/issues/2462
            logits = tf.layers.conv2d(net['conv0_up'], filters=n_class, kernel_size=1, padding='same')
    return logits, net


def Temporal_UNet(images, n_class, n_level, n_filter, n_block, training):
    """
        U-Net for segmenting an input image into n_class classes.

        images: NTXYC
        """
    with tf.variable_scope('Temporal_UNet'):
        net = {}
        x = images

        # Downsampling path
        # Learn fine-to-coarse features at each resolution level
        for l in range(0, n_level):
            with tf.variable_scope('conv{}'.format(l)):
                # If this is the first level (l = 0), keep the resolution.
                # Otherwise, convolve with a stride of 2, i.e. downsample by a
                # factor of 2.
                strides = 1 if l == 0 else 2
                # For each resolution level, perform n_block[l] times convolutions
                x = conv3d_bn_relu(x, filters=n_filter[l], training=training,
                                   kernel_size=3, strides=(1, strides, strides))
                for i in range(1, n_block[l]):
                    x = conv3d_bn_relu(x, filters=n_filter[l], training=training, kernel_size=3)
                net['conv{}'.format(l)] = x

        # Upsampling path
        l = n_level - 1
        with tf.variable_scope('conv{}_up'.format(l)):
            net['conv{}_up'.format(l)] = net['conv{}'.format(l)]

        for l in range(n_level - 2, -1, -1):
            with tf.variable_scope('conv{}_up'.format(l)):
                x = conv3d_transpose_bn_relu(net['conv{}_up'.format(l + 1)], filters=n_filter[l],
                                             training=training, kernel_size=3, strides=(1, 2, 2))
                x = tf.concat([net['conv{}'.format(l)], x], axis=-1)
                for i in range(0, n_block[l]):
                    x = conv3d_bn_relu(x, filters=n_filter[l], training=training, kernel_size=3)
                net['conv{}_up'.format(l)] = x

        # Perform prediction
        with tf.variable_scope('conv_out'):
            # We only calculate logits, instead of softmax here because the loss
            # function tf.nn.softmax_cross_entropy() accepts the unscaled logits
            # and performs softmax internally for efficiency and numerical
            # stability reasons.
            # Refer to https://github.com/tensorflow/tensorflow/issues/2462
            logits = tf.layers.conv3d(net['conv0_up'], filters=n_class, kernel_size=1, padding='same')
    return logits, net


def focal_loss(labels, logits, n_class, alpha):
    """ Focal loss """
    # labels: NXY
    # label_1hot: NXYC
    # logits: NXYC
    label_1hot = tf.one_hot(indices=labels, depth=n_class)

    # alpha_t: NXYC
    background = tf.equal(labels, 0)
    foreground = tf.not_equal(labels, 0)
    alpha_t = alpha * tf.cast(foreground, tf.float32) + (1 - alpha) * tf.cast(background, tf.float32)
    alpha_t = tf.expand_dims(alpha_t, axis=-1)

    # Multiply alpha_t with label_1hot first, then use
    # softmax_cross_entropy_with_logits
    label_1hot = alpha_t * label_1hot
    label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=label_1hot, logits=logits)
    loss = tf.reduce_mean(label_loss)
    return loss


def UNet_Model(images, labels, n_class, n_level, n_filter, n_block, training):
    """
        A model which takes input images, builds the graph and outputs the loss.

        images: NXYC
        labels: NXY
        """
    # Build the 2D U-Net model
    # images: NXYC
    # labels: NXY
    logits, net = UNet(images=images, n_class=n_class, n_level=n_level,
                       n_filter=n_filter, n_block=n_block, training=training)

    # The cross-entropy loss
    label_1hot = tf.one_hot(indices=labels, depth=n_class)
    label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=label_1hot, logits=logits)
    loss = tf.reduce_mean(label_loss)

    # The softmax probability and the predicted segmentation
    # prob: NXYC
    # pred: NXY
    prob = tf.nn.softmax(logits, name='prob')
    pred = tf.cast(tf.argmax(prob, axis=-1), dtype=tf.int32, name='pred')
    return loss, prob, pred


def Temporal_UNet_Model(images, labels, n_class, n_level, n_filter, n_block, n_step,
                        training, weight_R=1, weight_r=0):
    """
        A model which takes input images, builds the graph and outputs the loss.

        images: NTXYC
        labels: NTXY
        """
    # Build the 2D-t U-Net model
    # images: NTXYC
    # labels: NTXY
    logits, net = Temporal_UNet(images=images, n_class=n_class, n_level=n_level,
                                n_filter=n_filter, n_block=n_block, training=training)

    # Use all the time frames
    s = int((n_step - 1) / 2)
    loss = []
    sum_w = 0
    for t in range(n_step):
        # logits_fr: NXYC
        logits_fr = logits[:, t]
        label_1hot = tf.one_hot(indices=labels[:, t], depth=n_class)
        label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=label_1hot, logits=logits_fr)

        # weight
        d = abs(t - s)
        if d <= weight_R:
            w = pow(1 - float(d) / weight_R, weight_r)
        else:
            w = 0
        sum_w += w
        print(t, w)

        # loss
        # Curious: if w = 0 (boundary of the window), would this affect the stability
        # in loss gradient back-propagation?
        loss_fr = w * tf.reduce_mean(label_loss)
        loss += [loss_fr]

    # Average loss across time frames
    loss = tf.reduce_sum(tf.stack(loss, axis=0)) / sum_w

    # The softmax probability and the predicted segmentation
    # prob: NTXYC
    # pred: NTXY
    prob = tf.nn.softmax(logits, name='prob')
    pred = tf.cast(tf.argmax(prob, axis=-1), dtype=tf.int32, name='pred')
    return loss, prob, pred


def Conv_LSTM(features, lstm_input_shape, n_hidden, n_step, n_class):
    """
        Convolutional LSTM which processes a feature map.

        features: NTXYC
        input_shape: XYC, shape excluding the batch size
        n_hidden: dimension for both the hidden status and the output
        n_step:   number of unfolded temporal steps
        n_class:  number of label classes
        """
    with tf.variable_scope('LSTM'):
        cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=lstm_input_shape, output_channels=n_hidden,
                                             kernel_shape=[3, 3])

        initial_state = cell.zero_state(tf.shape(features)[0], tf.float32)
        state = initial_state

        # Simplified version of tensorflow_models/tutorials/rnn/rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=self._initial_state)
        outputs = []
        for t in range(n_step):
            if t > 0:
                tf.get_variable_scope().reuse_variables()
            # LSTM cell
            cell_output, state = cell(features[:, t], state)
            # Convolutional layer to match the number of classes
            logits = tf.layers.conv2d(cell_output, filters=n_class, kernel_size=1, padding='same', name='conv2d')
            # Concatenation of logits for each time step
            # outputs: a list of T tensors, each NXYC
            outputs.append(logits)
        # outputs: NTXYC
        outputs = tf.stack(outputs, axis=1)
    return outputs


def BiConv_LSTM(features, lstm_input_shape, n_hidden, n_step, n_class):
    """
        Bi-directional convolutional LSTM which processes a feature map.

        features: NTXYC
        input_shape: XYC, shape excluding the batch size
        n_hidden: dimension for both the hidden status and the output
        n_step:   number of unfolded temporal steps
        n_class:  number of label classes
        """
    with tf.variable_scope('LSTM'):
        # Simplified version of tensorflow_models/tutorials/rnn/rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
        #                            initial_state=self._initial_state)
        # Forward LSTM cell
        with tf.variable_scope('forward'):
            cell_fw = tf.contrib.rnn.Conv2DLSTMCell(input_shape=lstm_input_shape, output_channels=n_hidden,
                                                    kernel_shape=[3, 3])
            state_fw = cell_fw.zero_state(tf.shape(features)[0], tf.float32)
            cell_outputs_fw = []

            for t in range(n_step):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output_fw, state_fw = cell_fw(features[:, t], state_fw)
                cell_outputs_fw += [cell_output_fw]

        # Backward LSTM cell
        with tf.variable_scope('backward'):
            cell_bw = tf.contrib.rnn.Conv2DLSTMCell(input_shape=lstm_input_shape, output_channels=n_hidden,
                                                    kernel_shape=[3, 3])
            state_bw = cell_bw.zero_state(tf.shape(features)[0], tf.float32)
            cell_outputs_bw = []

            for t in range(n_step - 1, -1, -1):
                if t < n_step - 1:
                    tf.get_variable_scope().reuse_variables()
                cell_output_bw, state_bw = cell_bw(features[:, t], state_bw)
                cell_outputs_bw += [cell_output_bw]

        # Concatenate forward and backward outputs
        with tf.variable_scope('output'):
            outputs = []
            for t in range(n_step):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()

                # cell_output: NXYC
                cell_output = tf.concat([cell_outputs_fw[t], cell_outputs_bw[n_step - 1 - t]], axis=-1)

                # Convolutional layer to match the number of classes
                logits = tf.layers.conv2d(cell_output, filters=n_class, kernel_size=1, padding='same', name='conv2d')

                # Concatenation of logits for each time step
                # outputs: a list of T tensors, each NXYC
                outputs.append(logits)
            # outputs: NTXYC
            outputs = tf.stack(outputs, axis=1)
    return outputs


def UNet_LSTM_Model(images, labels, n_class, n_level, n_filter, n_block,
                    lstm_input_shape, n_hidden, n_step, training, training_UNet=False,
                    bidirectional=False, seq2seq=False, weight_R=1, weight_r=0):
    """
        A model which takes input images, builds the graph and outputs the loss.

        images: NTXYC
        labels: NTXY
        masks:  NTXY
        """
    # Merge the temporal dimension into the batch dimension
    images_shape = tf.shape(images)
    images = tf.reshape(images, [-1, images_shape[2], images_shape[3], images.shape[4]])

    # Generate the feature map using the UNet
    # images: (N*T)XYC
    _, net = UNet(images=images, n_class=n_class, n_level=n_level,
                  n_filter=n_filter, n_block=n_block, training=training_UNet)

    # features: (N*T)XYC
    features = net['conv0_up']

    # features: NTXYC
    features = tf.reshape(features, [images_shape[0], images_shape[1], images_shape[2], images_shape[3], n_filter[0]])

    # Pass the feature map to the LSTM
    # outputs: NTXYC
    if bidirectional:
        outputs = BiConv_LSTM(features, lstm_input_shape, n_hidden, n_step, n_class)
    else:
        outputs = Conv_LSTM(features, lstm_input_shape, n_hidden, n_step, n_class)

    if seq2seq:
        # Use all the time frames
        s = int((n_step - 1) / 2)
        loss = []
        sum_w = 0
        for t in range(n_step):
            # logits_fr: NXYC
            logits_fr = outputs[:, t]
            label_1hot = tf.one_hot(indices=labels[:, t], depth=n_class)
            label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=label_1hot, logits=logits_fr)

            # weight
            d = abs(t - s)
            if d <= weight_R:
                w = pow(1 - float(d) / weight_R, weight_r)
            else:
                w = 0
            sum_w += w
            print(t, w)

            # loss
            # Curious: if w = 0 (boundary of the window), would this affect the stability
            # in loss gradient back-propagation?
            loss_fr = w * tf.reduce_mean(label_loss)
            loss += [loss_fr]

        # Average loss across time frames
        loss = tf.reduce_sum(tf.stack(loss, axis=0)) / sum_w
    else:
        # Only focus on one time frame, where we have annotations.
        # The middle frame
        t_anno = int((n_step - 1) / 2)
        print(t_anno)

        # logits_fr: NXYC
        logits_fr = outputs[:, t_anno]
        label_1hot = tf.one_hot(indices=labels[:, t_anno], depth=n_class)
        label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=label_1hot, logits=logits_fr)
        loss = tf.reduce_mean(label_loss)

    # The softmax probability and the predicted segmentation
    # prob: NTXYC
    # pred: NTXY
    prob = tf.nn.softmax(outputs, name='prob')
    pred = tf.cast(tf.argmax(prob, axis=-1), dtype=tf.int32, name='pred')
    return loss, prob, pred
