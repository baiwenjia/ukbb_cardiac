# Copyright 2017, Wenjia Bai. All Rights Reserved.
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
import math
import numpy as np
import nibabel as nib
import tensorflow as tf
from ukbb_cardiac.common.image_utils import *


""" Deployment parameters """
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('time_step', 1,
                            'Time step during deployment of LSTM.')
tf.app.flags.DEFINE_enum('seq_name', 'ao', ['ao'],
                         'Sequence name.')
tf.app.flags.DEFINE_enum('model', 'UNet-LSTM', ['UNet', 'UNet-LSTM', 'Temporal-UNet'],
                         'Model name.')
tf.app.flags.DEFINE_string('data_dir',
                           'Biobank_ao/validation',
                           'Path to the test set directory, under which images '
                           'are organised in subdirectories for each subject.')
tf.app.flags.DEFINE_string('model_path',
                           'UNet-LSTM_ao_level5_filter16_22222_batch1_iter20000_lr0.001_zscore_tw9_h16_bidir_seq2seq_wR5_wr0.1_joint/UNet-LSTM_ao_level5_filter16_22222_batch1_iter20000_lr0.001_zscore_tw9_h16_bidir_seq2seq_wR5_wr0.1_joint.ckpt-20000',
                           'Path to the saved trained model.')
tf.app.flags.DEFINE_boolean('process_seq', True,
                            'Process a time sequence of images.')
tf.app.flags.DEFINE_boolean('save_seg', True,
                            'Save segmentation.')
tf.app.flags.DEFINE_boolean('z_score', True,
                            'Normalise the image intensity to z-score. '
                            'Otherwise, rescale the intensity.')
tf.app.flags.DEFINE_integer('weight_R', 5,
                            'Radius of the weighting window.')
tf.app.flags.DEFINE_float('weight_r', 0.1,
                          'Power of weight for the seq2seq loss. 0: uniform; 1: linear; 2: square.')


if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Import the computation graph and restore the variable values
        saver = tf.train.import_meta_graph('{0}.meta'.format(FLAGS.model_path))
        saver.restore(sess, '{0}'.format(FLAGS.model_path))

        print('Start evaluating on the test set ...')
        start_time = time.time()

        # Process each subject subdirectory
        data_list = sorted(os.listdir(FLAGS.data_dir))
        processed_list = []
        table = []
        for data in data_list:
            print(data)
            data_dir = os.path.join(FLAGS.data_dir, data)

            if FLAGS.process_seq:
                # Process the temporal sequence
                image_name = '{0}/{1}.nii.gz'.format(data_dir, FLAGS.seq_name)

                if not os.path.exists(image_name):
                    print('  Directory {0} does not contain an image with file name {1}. '
                          'Skip.'.format(data_dir, os.path.basename(image_name)))
                    continue

                # Read the image
                print('  Reading {} ...'.format(image_name))
                nim = nib.load(image_name)
                dx, dy, dz, dt = nim.header['pixdim'][1:5]
                area_per_pixel = dx * dy
                image = nim.get_data()
                X, Y, Z, T = image.shape
                orig_image = image

                print('  Segmenting full sequence ...')
                start_seg_time = time.time()

                # Intensity normalisation
                if FLAGS.z_score:
                    image = normalise_intensity(image, 10.0)
                else:
                    image = rescale_intensity(image, (1.0, 99.0))

                # Probability (segmentation)
                n_class = 3
                prob = np.zeros((X, Y, Z, T, n_class), dtype=np.float32)

                # Pad the image size to be a factor of 16 so that the downsample and upsample procedures
                # in the network will result in the same image size at each resolution level.
                # X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
                X2, Y2 = 256, 256
                x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
                x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
                image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0), (0, 0)), 'constant')

                # Process each time frame
                if FLAGS.model == 'UNet':
                    # For each time frame
                    for t in range(T):
                        # Transpose the shape to NXYC
                        image_fr = image[:, :, :, t]
                        image_fr = np.transpose(image_fr, axes=(2, 0, 1)).astype(np.float32)
                        image_fr = np.expand_dims(image_fr, axis=-1)

                        # Evaluate the network
                        # prob_fr: NXYC
                        prob_fr = sess.run('prob:0',
                                           feed_dict={'image:0': image_fr, 'training:0': False})

                        # Transpose and crop to recover the original size
                        # prob_fr: XYNC
                        prob_fr = np.transpose(prob_fr, axes=(1, 2, 0, 3))
                        prob_fr = prob_fr[x_pre:x_pre + X, y_pre:y_pre + Y]
                        prob[:, :, :, t, :] = prob_fr
                elif FLAGS.model == 'UNet-LSTM' or FLAGS.model == 'Temporal-UNet':
                    time_window = FLAGS.weight_R * 2 - 1
                    rad = int((time_window - 1) / 2)
                    weight = np.zeros((1, 1, 1, T, 1))

                    w = []
                    for t in range(time_window):
                        d = abs(t - rad)
                        if d <= FLAGS.weight_R:
                            w_t = pow(1 - float(d) / FLAGS.weight_R, FLAGS.weight_r)
                        else:
                            w_t = 0
                        w += [w_t]

                    w = np.array(w)
                    w = np.reshape(w, (1, 1, 1, time_window, 1))

                    # For each time frame after a time_step
                    for t in range(0, T, FLAGS.time_step):
                        # Get the frames in the time window
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

                        # image_idx: NTXYC
                        image_idx = image[:, :, :, idx]
                        image_idx = np.transpose(image_idx, axes=(2, 3, 0, 1)).astype(np.float32)
                        image_idx = np.expand_dims(image_idx, axis=-1)

                        # Evaluate the network
                        # Curious: can we deploy the LSTM model more efficiently by utilising the state variable?
                        # Currently, we have to feed all the time frames in the time window and we can not just
                        # feed one time frame, because the LSTM is an unrolled model in the dataflow graph.
                        # It needs all the input from the time window.
                        # prob_idx: NTXYC
                        prob_idx = sess.run('prob:0',
                                            feed_dict={'image:0': image_idx, 'training:0': False})

                        # Transpose and crop the segmentation to recover the original size
                        # prob_idx: XYNTC
                        prob_idx = np.transpose(prob_idx, axes=(2, 3, 0, 1, 4))

                        # Tile the overlapping probability maps
                        prob[:, :, :, idx] += prob_idx[x_pre:x_pre + X, y_pre:y_pre + Y] * w
                        weight[:, :, :, idx] += w

                    # Average probability
                    prob /= weight
                else:
                    print('Error: unknown model {0}.'.format(FLAGS.model))
                    exit(0)

                # Segmentation
                pred = np.argmax(prob, axis=-1).astype(np.int32)

                # Save the segmentation
                if FLAGS.save_seg:
                    print('  Saving segmentation ...')
                    nim2 = nib.Nifti1Image(pred, nim.affine)
                    nim2.header['pixdim'] = nim.header['pixdim']
                    nib.save(nim2, '{0}/seg_{1}.nii.gz'.format(data_dir, FLAGS.seq_name))

                seg_time = time.time() - start_seg_time
                print('  Segmentation time = {:3f}s'.format(seg_time))
                processed_list += [data]
            else:
                if FLAGS.model == 'UNet-LSTM':
                    print('UNet-LSTM does not support frame-wise segmentation. '
                          'Please use the -process_seq flag.')
                    exit(0)

                # Process ED and ES time frames
                image_ED_name = '{0}/{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, 'ED')
                image_ES_name = '{0}/{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, 'ES')
                if not os.path.exists(image_ED_name) or not os.path.exists(image_ES_name):
                    print('  Directory {0} does not contain an image with file name {1} or {2}. '
                          'Skip.'.format(data_dir, os.path.basename(image_ED_name), os.path.basename(image_ES_name)))
                    continue

                measure = {}
                for fr in ['ED', 'ES']:
                    image_name = '{0}/{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, fr)

                    # Read the image
                    # image: XYZ
                    print('  Reading {} ...'.format(image_name))
                    nim = nib.load(image_name)
                    dx, dy, dz, dt = nim.header['pixdim'][1:5]
                    area_per_pixel = dx * dy
                    image = nim.get_data()
                    X, Y = image.shape[:2]

                    print('  Segmenting {} frame ...'.format(fr))
                    start_seg_time = time.time()

                    # Intensity normalisation
                    if FLAGS.z_score:
                        image = normalise_intensity(image, 10.0)
                    else:
                        image = rescale_intensity(image, (1.0, 99.0))

                    # Pad the image size to be a factor of 16 so that the downsample and upsample procedures
                    # in the network will result in the same image size at each resolution level.
                    X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
                    x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
                    x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
                    image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0)), 'constant')

                    # Transpose the shape to NXYC
                    # image: NXY
                    image = np.transpose(image, axes=(2, 0, 1)).astype(np.float32)
                    # image: NXYC
                    image = np.expand_dims(image, axis=-1)

                    # Evaluate the network
                    # pred: NXY
                    prob, pred = sess.run(['prob:0', 'pred:0'],
                                          feed_dict={'image:0': image, 'training:0': False})

                    # Transpose and crop the segmentation to recover the original size
                    pred = np.transpose(pred, axes=(1, 2, 0))
                    pred = pred[x_pre:x_pre + X, y_pre:y_pre + Y]

                    seg_time = time.time() - start_seg_time
                    print('  Segmentation time = {:3f}s'.format(seg_time))

                    # Save the segmentation
                    if FLAGS.save_seg:
                        print('  Saving segmentation ...')
                        nim2 = nib.Nifti1Image(pred, nim.affine)
                        nim2.header['pixdim'] = nim.header['pixdim']
                        nib.save(nim2, '{0}/seg_{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, fr))

                processed_list += [data]

        process_time = time.time() - start_time
        print('Including image I/O, CUDA resource allocation, '
              'it took {:.3f}s for processing {:d} subjects ({:.3f}s per subjects).'.format(
            process_time, len(processed_list), process_time / len(processed_list)))
