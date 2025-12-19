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
import os
import time
import random
import numpy as np
import nibabel as nib
import tensorflow as tf
import cv2
from ukbb_cardiac.common.network import *
from ukbb_cardiac.common.network_ao import *
from ukbb_cardiac.common.image_utils import *


""" Training parameters """
FLAGS = tf.app.flags.FLAGS
# NOTE: use image_size = 256 for aortic images to learn the boundary.
# Otherwise, the boundary may be misunderstood as the aorta.
tf.app.flags.DEFINE_integer('image_size', 256,
                            'Image size after cropping.')
tf.app.flags.DEFINE_integer('time_window', 11,
                            'Time window for LSTM.')
tf.app.flags.DEFINE_integer('train_batch_size', 5,
                            'Number of images for each training batch.')
tf.app.flags.DEFINE_integer('validation_batch_size', 5,
                            'Number of images for each validation batch.')
tf.app.flags.DEFINE_integer('num_filter', 16,
                            'Number of filters for the first convolution layer.')
tf.app.flags.DEFINE_integer('num_level', 5,
                            'Number of network levels.')
tf.app.flags.DEFINE_integer('num_hidden', 16,
                            'Number of hidden status dimension in LSTM.')
tf.app.flags.DEFINE_integer('train_iteration', 20000,
                            'Number of training iterations.')
tf.app.flags.DEFINE_float('learning_rate', 1e-3,
                          'Learning rate.')
tf.app.flags._global_parser.add_argument('--reduce_lr_after', action='append',
                                         help='Reduce the learning rate after this many iterations.')
tf.app.flags.DEFINE_enum('seq_name', 'ao', ['ao'],
                         'Sequence name.')
tf.app.flags.DEFINE_enum('model', 'UNet', ['UNet', 'UNet-LSTM', 'Temporal-UNet'],
                         'Model name.')
tf.app.flags.DEFINE_string('dataset_dir',
                           'Biobank_ao',
                           'Path to the dataset directory, which is split into '
                           'training and validation subdirectories.')
tf.app.flags.DEFINE_string('log_dir', 'log',
                           'Directory for saving the log file.')
tf.app.flags.DEFINE_string('checkpoint_dir', 'model',
                           'Directory for saving the trained model.')
tf.app.flags.DEFINE_string('model_path', '',
                           'Path to the saved trained model.')
tf.app.flags.DEFINE_boolean('z_score', True,
                            'Normalise the image intensity to z-score. Otherwise, rescale the intensity.')
tf.app.flags.DEFINE_boolean('bidirectional', True,
                            'Bi-directional LSTM.')
tf.app.flags.DEFINE_boolean('seq2seq', True,
                            'Sequence to sequence learning. Otherwise, only learn the last time frame.')
tf.app.flags.DEFINE_integer('weight_R', 5,
                            'Radius of the weighting window.')
tf.app.flags.DEFINE_float('weight_r', 0,
                          'Power of weight for the seq2seq loss. 0: uniform; 1: linear; 2: square.')
tf.app.flags.DEFINE_boolean('joint_train', False,
                            'Joint training of UNet and LSTM.')
tf.app.flags.DEFINE_boolean('from_scratch', False,
                            'Train from scratch for UNet-LSTM.')


def get_trusted_mask(label_map, radius=5):
    """ Get a trusted mask region from the label map at another time frame"""
    # Binary foreground and background
    fg = (label_map > 0)
    bg = ~fg

    # Erosion of foreground and background
    size = radius * 2 - 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    fg2 = cv2.erode(fg.astype(np.uint8), kernel)
    bg2 = cv2.erode(bg.astype(np.uint8), kernel)

    # Union of eroded foreground and background
    mask = np.logical_or(fg2, bg2).astype(np.int8)
    return mask


def get_random_batch(filename_list, batch_size, image_size=192, time_window=1, data_augmentation=False,
                     shift=0.0, rotate=0.0, scale=0.0, intensity=0.0, flip=False):
    # Randomly select batch_size images from filename_list
    n_file = len(filename_list)
    n_selected = 0
    images = []
    labels = []
    while n_selected < batch_size:
        rand_index = random.randrange(n_file)
        if len(filename_list[rand_index]) == 2:
            image_name, label_name = filename_list[rand_index]
            label_prop_name = None
        else:
            image_name, label_name, label_prop_name = filename_list[rand_index]
        if os.path.exists(image_name) and os.path.exists(label_name):
            print('  Select {0} {1}'.format(image_name, label_name))

            # Read image and label
            image = nib.load(image_name).get_data()
            label = nib.load(label_name).get_data()

            # label is a temporally sparse annotation
            # label_prop has annotation across all time frames
            if label_prop_name:
                label_prop = nib.load(label_prop_name).get_data()
            else:
                label_prop = None

            # Handle exceptions
            if image.shape != label.shape:
                print('Error: mismatched size, image.shape = {0}, label.shape = {1}'.format(
                    image.shape, label.shape))
                print('Skip {0}, {1}'.format(image_name, label_name))
                continue

            if label_prop_name:
                if image.shape != label_prop.shape:
                    print('Error: mismatched size, image.shape = {0}, label_prop.shape = {1}'.format(
                        image.shape, label.shape))
                    print('Skip {0}, {1}'.format(image_name, label_name))
                    continue

            if image.max() < 1e-6:
                print('Error: blank image, image.max = {0}'.format(image.max()))
                print('Skip {0} {1}'.format(image_name, label_name))
                continue

            # Normalise the image size
            X, Y, Z, T = image.shape
            cx, cy = int(X / 2), int(Y / 2)
            image = crop_image(image, cx, cy, image_size)
            label = crop_image(label, cx, cy, image_size)
            if label_prop_name:
                label_prop = crop_image(label_prop, cx, cy, image_size)

            # Intensity normalisation
            if FLAGS.z_score:
                image = normalise_intensity(image, 10.0)
            else:
                image = rescale_intensity(image, (1.0, 99.0))

            # Get the time frames with annotations
            t_anno = np.nonzero(np.sum((label > 0), axis=(0, 1, 2)))[0]

            # For each annotated time frame, get a time window centred at here
            for t in t_anno:
                rad = int((time_window - 1) / 2)
                t1 = t - rad
                t2 = t + rad
                idx = []
                for i in range(t1, t2 + 1):
                    if i < 0:
                        idx += [i + T]
                    elif i >= T:
                        idx += [i - T]
                    else:
                        idx += [i]
                print(t, idx)

                # image: TXY
                image_idx = np.transpose(image[:, :, 0, idx], (2, 0, 1))

                # label: TXY
                if label_prop_name:
                    # print(label_prop_name)
                    label_idx = np.transpose(label_prop[:, :, 0, idx], (2, 0, 1))
                else:
                    # If there is no annotation across the time frames, simply
                    # copy the central time frame to other frames
                    label_idx = np.repeat(np.expand_dims(label[:, :, 0, t], axis=0), time_window, axis=0)

                # Add the channel dimension
                # image: TXYC
                image_idx = np.expand_dims(image_idx, axis=-1)

                # Perform data augmentation
                if data_augmentation:
                    image_idx, label_idx = aortic_data_augmenter(image_idx, label_idx, shift=shift, rotate=rotate,
                                                                 scale=scale, intensity=intensity, flip=flip)

                images += [image_idx]
                labels += [label_idx]

            # Increase the counter
            n_selected += 1

    # Convert to a numpy array
    # Default shape:
    # images: NTXYC
    # labels: NTXY
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    if FLAGS.model == 'UNet':
        # images: NXYC
        # labels: NXY
        images = np.reshape(images, (-1, images.shape[2], images.shape[3], images.shape[4]))
        labels = np.reshape(labels, (-1, labels.shape[2], labels.shape[3]))
    return images, labels


def main(argv=None):
    """ Main function """
    # Go through each subset (training, validation) under the data directory
    # and list the file names of the subjects
    data_list = {}
    for k in ['train', 'validation']:
        subset_dir = os.path.join(FLAGS.dataset_dir, k)
        data_list[k] = []
        for data in sorted(os.listdir(subset_dir)):
            data_dir = os.path.join(subset_dir, data)
            # Check the existence of the image and label map
            # and add their file names to the list
            image_name = '{0}/{1}.nii.gz'.format(data_dir, FLAGS.seq_name)
            label_name = '{0}/label_{1}.nii.gz'.format(data_dir, FLAGS.seq_name)
            label_prop_name = '{0}/label_{1}_prop.nii.gz'.format(data_dir, FLAGS.seq_name)
            if os.path.exists(image_name) and os.path.exists(label_name):
                if os.path.exists(label_prop_name):
                    data_list[k] += [[image_name, label_name, label_prop_name]]
                else:
                    data_list[k] += [[image_name, label_name]]

    # Prepare tensors for the image and label map pairs
    # Use int32 for label_pl as tf.one_hot uses int32
    if FLAGS.model == 'UNet':
        # image_pl: NXYC
        # label_pl: NXY
        image_pl = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='image')
        label_pl = tf.placeholder(tf.int32, shape=[None, None, None], name='label')
    elif FLAGS.model == 'UNet-LSTM' or FLAGS.model == 'Temporal-UNet':
        # image_pl: NTXYC
        # label_pl: NTXY
        # mask_pl:  NTXY
        image_pl = tf.placeholder(tf.float32, shape=[None, None, None, None, 1], name='image')
        label_pl = tf.placeholder(tf.int32, shape=[None, None, None, None], name='label')
    else:
        print('Error: unknown model {0}.'.format(FLAGS.model))
        exit(0)

    # Placeholder for the training phase
    # This flag is important for the batch_normalization layer to function
    # properly.
    training_pl = tf.placeholder(tf.bool, shape=[], name='training')

    # Determine the number of label classes according to the manual annotation
    # procedure for each image sequence.
    n_class = 0
    if FLAGS.seq_name == 'ao':
        # ao, aortic distensibility images
        # 3 classes (background, ascending aorta, descending aorta)
        n_class = 3
    else:
        print('Error: unknown seq_name {0}.'.format(FLAGS.seq_name))
        exit(0)

    # The number of filters at each resolution level
    # Follow the VGG philosophy, increasing the dimension by a factor of 2 for each level
    n_filter = []
    for i in range(FLAGS.num_level):
        n_filter += [FLAGS.num_filter * pow(2, i)]
    print('Number of filters at each level =', n_filter)
    print('Note: The connection between neurons is proportional to n_filter * n_filter. '
          'Increasing n_filter by a factor of 2 will increase the number of parameters by a factor of 4. '
          'So it is better to start experiments with a small n_filter and increase it later.')

    # Build the neural network model
    n_block = [2, 2, 2, 2, 2]
    if FLAGS.model == 'UNet':
        loss, prob, pred = UNet_Model(image_pl, label_pl, n_class, FLAGS.num_level,
                                      n_filter, n_block, training_pl)
        time_window = 1
        label_fr = label_pl
        pred_fr = pred
    elif FLAGS.model == 'UNet-LSTM':
        lstm_input_shape = [FLAGS.image_size, FLAGS.image_size, n_filter[0]]
        # The unrolled time window is determined by the weighting window radius
        time_window = FLAGS.weight_R * 2 - 1
        loss, prob, pred = UNet_LSTM_Model(image_pl, label_pl, n_class,
                                           FLAGS.num_level, n_filter, n_block,
                                           lstm_input_shape, FLAGS.num_hidden,
                                           time_window,
                                           training_pl, training_UNet=False,
                                           bidirectional=FLAGS.bidirectional,
                                           seq2seq=FLAGS.seq2seq,
                                           weight_R=FLAGS.weight_R,
                                           weight_r=FLAGS.weight_r)
        s = int((time_window - 1) / 2)
        label_fr = label_pl[:, s]
        pred_fr = pred[:, s]
    elif FLAGS.model == 'Temporal-UNet':
        time_window = FLAGS.weight_R * 2 - 1
        loss, prob, pred = Temporal_UNet_Model(image_pl, label_pl, n_class,
                                               FLAGS.num_level, n_filter, n_block,
                                               time_window,
                                               training_pl,
                                               weight_R=FLAGS.weight_R,
                                               weight_r=FLAGS.weight_r)
        s = int((time_window - 1) / 2)
        label_fr = label_pl[:, s]
        pred_fr = pred[:, s]
    else:
        print('Error: unknown model {0}.'.format(FLAGS.model))
        exit(0)

    # Evaluation metrics on the annotated frame
    accuracy = tf_categorical_accuracy(pred_fr, label_fr)
    dice_aa = tf_categorical_dice(pred_fr, label_fr, 1)
    dice_da = tf_categorical_dice(pred_fr, label_fr, 2)

    # Learning rate
    global_step = tf.Variable(0, trainable=False)
    if FLAGS.reduce_lr_after:
        boundaries = [int(x) for x in FLAGS.reduce_lr_after]
        boundaries = sorted(boundaries)
        values = []
        val = FLAGS.learning_rate
        for i in range(len(boundaries) + 1):
            values += [val]
            val *= 0.1
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
    else:
        learning_rate = FLAGS.learning_rate

    # Optimiser
    optimizer = tf.train.AdamOptimizer(learning_rate)
    if FLAGS.model == 'UNet' or FLAGS.model == 'Temporal-UNet':
        # We need to add the operators associated with batch_normalization to
        # the optimiser, according to
        # https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)
    elif FLAGS.model == 'UNet-LSTM':
        if not FLAGS.joint_train:
            # Only train the LSTM part
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'LSTM/')
            with tf.control_dependencies(update_ops):
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'LSTM/')
                print(var_list)
                train_op = optimizer.minimize(loss, var_list=var_list, global_step=global_step)
        else:
            # Jointly train the UNet and LSTM parts
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=global_step)

    # Model name and directory
    model_name = '{0}_{1}_level{2}_filter{3}_{4}_batch{5}_iter{6}_lr{7}'.format(
        FLAGS.model, FLAGS.seq_name, FLAGS.num_level, n_filter[0], ''.join([str(x) for x in n_block]),
        FLAGS.train_batch_size, FLAGS.train_iteration, FLAGS.learning_rate)
    if FLAGS.z_score:
        model_name += '_zscore'
    if FLAGS.model == 'UNet-LSTM':
        model_name += '_tw{0}_h{1}'.format(time_window, FLAGS.num_hidden)
        if FLAGS.bidirectional:
            model_name += '_bidir'
        if FLAGS.seq2seq:
            model_name += '_seq2seq_wR{0}_wr{1}'.format(FLAGS.weight_R, FLAGS.weight_r)
        if FLAGS.joint_train:
            model_name += '_joint'
        if FLAGS.from_scratch:
            model_name += '_scratch'
    if FLAGS.model == 'Temporal-UNet':
        model_name += '_tw{0}_wR{1}_wr{2}'.format(time_window, FLAGS.weight_R, FLAGS.weight_r)
    print(model_name)

    model_dir = os.path.join(FLAGS.checkpoint_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Start the tensorflow session
    with tf.Session() as sess:
        print('Start training...')
        start_time = time.time()

        # Create a saver
        saver = tf.train.Saver(max_to_keep=20)

        # Summary writer
        summary_dir = os.path.join(FLAGS.log_dir, model_name)
        if os.path.exists(summary_dir):
            os.system('rm -rf {0}'.format(summary_dir))
        train_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'train'), graph=sess.graph)
        val_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'validation'), graph=sess.graph)

        # Initialise variables
        sess.run(tf.global_variables_initializer())

        # Import the pre-trained U-Net weights if not training from scratch
        if FLAGS.model == 'UNet-LSTM' and not FLAGS.from_scratch:
            # Important, restore all the GLOBAL_VARIABLES here.
            # If using TRAINABLE_VARIABLES, the moving_mean and moving_variance for
            # batch_normalisation will not be included.
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'UNet/')
            print('Restore pre-trained UNet weights...')
            saver2 = tf.train.Saver(var_list)
            saver2.restore(sess, '{0}'.format(FLAGS.model_path))

        # Iterate
        for iteration in range(1, 1 + FLAGS.train_iteration):
            # For each iteration, we randomly choose a batch of subjects for training
            print('Iteration {0}: training...'.format(iteration))
            start_time_iter = time.time()

            images, labels = get_random_batch(data_list['train'],
                                              FLAGS.train_batch_size,
                                              image_size=FLAGS.image_size,
                                              time_window=time_window,
                                              data_augmentation=True,
                                              shift=0, rotate=10, scale=0.1,
                                              intensity=0, flip=False)

            # Stochastic optimisation using this batch
            _, train_loss, train_acc, lr = sess.run([train_op, loss, accuracy, learning_rate],
                                                    {image_pl: images, label_pl: labels, training_pl: True})

            summary = tf.Summary()
            summary.value.add(tag='loss', simple_value=train_loss)
            summary.value.add(tag='accuracy', simple_value=train_acc)
            summary.value.add(tag='learning_rate', simple_value=lr)
            train_writer.add_summary(summary, iteration)

            # After every ten iterations, we perform validation
            if iteration % 10 == 0:
                print('Iteration {0}: validation...'.format(iteration))
                images, labels = get_random_batch(data_list['validation'],
                                                  FLAGS.validation_batch_size,
                                                  image_size=FLAGS.image_size,
                                                  time_window=time_window,
                                                  data_augmentation=False)

                val_loss, val_acc, val_dice_aa, val_dice_da = sess.run([loss, accuracy, dice_aa, dice_da],
                                                                       {image_pl: images, label_pl: labels, training_pl: False})

                summary = tf.Summary()
                summary.value.add(tag='loss', simple_value=val_loss)
                summary.value.add(tag='accuracy', simple_value=val_acc)
                summary.value.add(tag='dice_aa', simple_value=val_dice_aa)
                summary.value.add(tag='dice_da', simple_value=val_dice_da)
                val_writer.add_summary(summary, iteration)

                # Print the results for this iteration
                print('Iteration {} of {} took {:.3f}s'.format(iteration, FLAGS.train_iteration,
                                                               time.time() - start_time_iter))
                print('  training loss:\t\t{:.6f}'.format(train_loss))
                print('  training accuracy:\t\t{:.2f}%'.format(train_acc * 100))
                print('  validation loss: \t\t{:.6f}'.format(val_loss))
                print('  validation accuracy:\t\t{:.2f}%'.format(val_acc * 100))
                print('  validation Dice AA:\t\t{:.6f}'.format(val_dice_aa))
                print('  validation Dice DA:\t\t{:.6f}'.format(val_dice_da))
            else:
                # Print the results for this iteration
                print('Iteration {} of {} took {:.3f}s'.format(
                    iteration, FLAGS.train_iteration, time.time() - start_time_iter))
                print('  training loss:\t\t{:.6f}'.format(train_loss))
                print('  training accuracy:\t\t{:.2f}%'.format(train_acc * 100))

            # Save models after every 1000 iterations
            if iteration % 1000 == 0:
                saver.save(sess, save_path=os.path.join(model_dir, '{0}.ckpt'.format(model_name)),
                           global_step=iteration)

        # Close the summary writers
        train_writer.close()
        val_writer.close()
        print('Training took {:.3f}s in total.\n'.format(time.time() - start_time))


if __name__ == '__main__':
    tf.app.run()
