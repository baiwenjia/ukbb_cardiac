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
"""
    Prepare the training, validation and test data for UK Biobank Application 2964.

    This script first performs quality control for the images and manual annotations,
    then splits the data into training/validation/test so that we could train and
    evaluate our neural networks.
    """
import os
import random
import numpy as np
import pandas as pd
import nibabel as nib


if __name__ == '__main__':
    # Path to the downloaded data
    orig_path = 'downloaded_data_path'

    # Path to the cleaned and prepared data
    data_path = 'data_path'

    # Where the csv spreadsheet is saved
    csv_path = 'csv_path'

    """
        Step 1: filter out the subjects with manual annotations.
        """
    # For each subdirectory
    for sub_path in sorted(os.listdir(orig_path)):
        sub_path = os.path.join(orig_path, sub_path)
        # For each subject in the subdirectory
        for eid in sorted(os.listdir(sub_path)):
            print(eid)
            orig_dir = os.path.join(sub_path, eid)
            if os.path.exists('{0}/{1}_cvi42.zip'.format(orig_dir, eid)):
                data_dir = os.path.join(data_path, eid)
                if not os.path.exists(data_dir):
                    os.mkdir(data_dir)

                # Create symbolic links in the prepared data directory
                for seq in ['sa', 'la_2ch', 'la_4ch']:
                    if os.path.exists('{0}/{1}.nii.gz'.format(orig_dir, seq)) \
                            and os.path.exists('{0}/label_{1}.nii.gz'.format(orig_dir, seq))\
                            and os.path.exists('{0}/label_up_{1}.nii.gz'.format(orig_dir, seq)):
                        os.system('ln -sf {0}/{1}.nii.gz {2}'.format(orig_dir, seq, data_dir))
                        os.system('ln -sf {0}/label_{1}.nii.gz {2}'.format(orig_dir, seq, data_dir))
                        os.system('ln -sf {0}/label_up_{1}.nii.gz {2}'.format(orig_dir, seq, data_dir))

                # Remove empty folders
                if len(os.listdir(data_dir)) == 0:
                    os.rmdir(data_dir)

    """
        Step 2: Extract the ED and ES time frames for the image and the label map if
        both time frames have been annotated.
        """
    for seq in ['sa', 'la_2ch', 'la_4ch']:
        for eid in sorted(os.listdir(data_path)):
            print(eid)
            data_dir = os.path.join(data_path, eid)
            image_name = '{0}/{1}.nii.gz'.format(data_dir, seq)
            label_name = '{0}/label_{1}.nii.gz'.format(data_dir, seq)
            label_up_name = '{0}/label_up_{1}.nii.gz'.format(data_dir, seq)
            if os.path.exists(label_name):
                nim = nib.load(label_name)
                label = nim.get_data()

                # Check the annotation across time frames
                proj_t = np.sum(label, axis=(0, 1, 2))
                index_t = []

                # Check whether the annotation contains all the label classes.
                # For example, for short-axis images, sometimes the RV is not annotated.
                for t in np.nonzero(proj_t)[0]:
                    if seq == 'sa':
                        if np.array_equal(np.unique(label[:, :, :, t]), [0, 1, 2, 3]):
                            index_t += [t]
                    elif seq == 'la_2ch':
                        if np.array_equal(np.unique(label[:, :, :, t]), [0, 1]):
                            index_t += [t]
                    elif seq == 'la_4ch':
                        if np.array_equal(np.unique(label[:, :, :, t]), [0, 1, 2]):
                            index_t += [t]

                # If there is valid annotation at 3 time frames or more, i.e. several ES annotations,
                # choose the one with the smallest volume for short-axis images and largest volume
                # for long-axis images
                if len(index_t) >= 3:
                    index_ES = index_t[1:]
                    if seq == 'sa':
                        index_t = [index_t[0], index_ES[np.argmin(proj_t[index_ES])]]
                    else:
                        index_t = [index_t[0], index_ES[np.argmax(proj_t[index_ES])]]

                if proj_t[0] == 0:
                    print('  Error at {0}: no annotation at ED frame.'.format(label_name))
                    continue
                if len(index_t) == 1:
                    print('  Error at {0}: annotation only available at one frame.'.format(label_name))
                    continue
                if len(index_t) != 2:
                    print('  Error at {0}: annotation not available for two frames.'.format(label_name))
                    continue

                # The ED and ES time frames
                fr = {}
                fr['ED'] = index_t[0]
                fr['ES'] = index_t[1]

                # Save the image and label map for ED and ES frames
                nim = nib.load(image_name)
                vol = nim.get_data()

                nim_up = nib.load(label_up_name)
                label_up = nim_up.get_data()

                for k, v in fr.items():
                    nib.save(nib.Nifti1Image(vol[:, :, :, v], nim.affine),
                             '{0}/{1}_{2}.nii.gz'.format(data_dir, seq, k))
                    nib.save(nib.Nifti1Image(label[:, :, :, v], nim.affine),
                             '{0}/label_{1}_{2}.nii.gz'.format(data_dir, seq, k))
                    nib.save(nib.Nifti1Image(label_up[:, :, :, v], nim_up.affine),
                             '{0}/label_up_{1}_{2}.nii.gz'.format(data_dir, seq, k))

    """
        Step 3: Perform further quality control. Check the following conditions:
        1. the image is not black
        2. the annotation contains all the classes (sometimes the RV is not annotated)
        """
    for seq in ['sa', 'la_2ch', 'la_4ch']:
        good_data = []

        for eid in sorted(os.listdir(data_path)):
            data_dir = os.path.join(data_path, eid)
            flag_good = True
            for fr in ['ED', 'ES']:
                # Check whether the image exists
                image_name = '{0}/{1}_{2}.nii.gz'.format(data_dir, seq, fr)
                if not os.path.exists(image_name):
                    flag_good = False
                    break

                # Check whether the image is black or not
                image = nib.load(image_name).get_data()
                if image.max() < 1e-6:
                    flag_good = False
                    break

                # Check whether the annotation exists
                label_name = '{0}/label_{1}_{2}.nii.gz'.format(data_dir, seq, fr)
                if not os.path.exists(label_name):
                    flag_good = False
                    break

            if flag_good:
                # Change to integer in order to perform set difference operation with bad_data list
                good_data += [int(data)]

        # There are some subjects that only miss one or several slices of annotations.
        # They were manually spotted out by checking the errors in clinical measures.
        bad_data = pd.read_csv(os.path.join(csv_path, 'bad_eid_{0}.csv'.format(seq)))['eid'].tolist()
        good_data = sorted(list(set(good_data) - set(bad_data)))

        # Save the list after quality control
        df = pd.DataFrame(good_data, index=None, columns=['eid'])
        df.to_csv(os.path.join(csv_path, 'good_eid_{0}.csv'.format(seq)), index=None)

    """
        Step 4: Split the dataset into training, validation and testing sets.
        """
    for seq in ['sa', 'la_2ch', 'la_4ch']:
        data_list = pd.read_csv(os.path.join(csv_path, 'good_eid_{0}.csv'.format(seq)))['eid'].tolist()
        n_data = len(data_list)

        # The ICC list for evaluating human inter-observer variability
        icc_list = np.unique(pd.read_csv('icc_atrial_data_50cases.csv', index_col=0).index)

        # Avoid the ICC subjects from being included into the training set.
        # We will keep these subjects in the test set.
        icc_list = sorted(list(set(icc_list) - (set(icc_list) - set(data_list))))
        n_icc = len(icc_list)
        data_list_wo_icc = sorted(list(set(data_list) - set(icc_list)))

        # Random shuffle the list
        random.shuffle(data_list_wo_icc)
        n_validation = 300
        n_test = 600
        n_test_wo_icc = n_test - n_icc
        n_train = n_data - n_validation - n_test

        # Create the list for training, validation and test
        sub_list = {}
        sub_list['train'] = data_list_wo_icc[:n_train]
        sub_list['validation'] = data_list_wo_icc[n_train:n_train + n_validation]
        sub_list['test'] = data_list_wo_icc[n_train + n_validation:] + list(icc_list)

        for c in ['train', 'validation', 'test']:
            pd.DataFrame(sub_list[c], columns=['eid']).to_csv(os.path.join(csv_path, '{}_{}.csv'.format(seq, c)), index=None)

        # Create directories for training, validation and test
        for k, v in sub_list.items():
            sub_dir = os.path.join('some_data_path', seq, k)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

            for eid in v:
                data_dir = os.path.join(data_path, str(eid))
                dest_dir = os.path.join(sub_dir, str(eid))
                os.system('ln -s {0} {1}'.format(data_dir, dest_dir))
