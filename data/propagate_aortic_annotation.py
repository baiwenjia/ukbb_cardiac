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
"""
    Propagate manual annotations at ED and ES frames to other time frames using image registration.
    """
import os
import sys
import nibabel as nib
import numpy as np
from ukbb_cardiac.common.image_utils import *


def infer_time_frame(image_name, image_fr_name):
    """ Infer which time frame the annotation is at. """
    nim = nib.load(image_name)
    T = nim.header['dim'][4]
    image = nim.get_data()
    nim_fr = nib.load(image_fr_name)
    image_fr = nim_fr.get_data()

    diff = np.zeros(T)
    for t in range(T):
        diff[t] = np.sum(np.abs(image[:, :, :, t] - image_fr))
    k = np.argmin(diff)
    return k


def wrap_frame_index(t_index, T):
    """ Handle frame index if it less than 0 or larger than T. """
    t2_index = []
    for t in t_index:
        if t < 0:
            t2 = t + T
        elif t >= T:
            t2 = t - T
        else:
            t2 = t
        t2_index += [t2]
    return t2_index


if __name__ == '__main__':
    data_path = 'Biobank_ao_data'
    data_list = sorted(os.listdir(data_path))
    par_path = 'ukbb_cardiac/par'

    for data in data_list:
        print(data)
        data_dir = os.path.join(data_path, data)

        # Directory for motion tracking results
        motion_dir = os.path.join(data_dir, 'motion')
        if not os.path.exists(motion_dir):
            os.mkdir(motion_dir)

        # Split the image sequence
        image_name = '{0}/ao.nii.gz'.format(data_dir)
        output_name = '{0}/ao_fr'.format(motion_dir)
        split_sequence(image_name, output_name)

        # The number of time frames
        nim = nib.load(image_name)
        T = nim.header['dim'][4]
        dt = nim.header['pixdim'][4]

        # Get the index of ED and ES time frames
        t_anno = []
        for fr in ['ED', 'ES']:
            image_fr_name = '{0}/ao_{1}.nii.gz'.format(data_dir, fr)
            k = infer_time_frame(image_name, image_fr_name)
            t_anno += [k]

            # Copy the annotation if it has been annotated
            os.system('cp {0}/label_ao_{1}.nii.gz {2}/label_ao_prop{3:02d}.nii.gz'.format(data_dir, fr, motion_dir, k))

        # Get the ROI for image registration by cropping the annotation
        auto_crop_image('{0}/label_ao.nii.gz'.format(data_dir), '{0}/label_ao_crop.nii.gz'.format(motion_dir), 10)
        os.system('mirtk transform-image {0}/ao.nii.gz {1}/ao_crop.nii.gz '
                  '-target {1}/label_ao_crop.nii.gz'.format(data_dir, motion_dir))

        # Split the cropped image sequence
        split_sequence('{0}/ao_crop.nii.gz'.format(motion_dir), '{0}/ao_crop_fr'.format(motion_dir))

        # Prepare for ED and ES annotation propagation
        prop_idx = {}
        for t in t_anno:
            prop_idx[t] = {}
            prop_idx[t]['forward'] = []
            prop_idx[t]['backward'] = []

        # For un-annotated frames, find its closest time frame
        for t in range(T):
            if t in t_anno:
                continue

            dist = np.abs(t - np.array(t_anno))
            dist = [x if (x <= T / 2) else (T - x) for x in dist]
            source_t = t_anno[np.argmin(dist)]

            # Determine whether it is forward or backward propagation
            d = t - source_t
            if d > T / 2:
                prop_idx[source_t]['backward'] += [t]
            elif d > 0:
                prop_idx[source_t]['forward'] += [t]
            elif d > - T / 2:
                prop_idx[source_t]['backward'] += [t]
            else:
                prop_idx[source_t]['forward'] += [t]

        # Sort the propagation order and propagate closer frames first
        for t in t_anno:
            for dir in ['forward', 'backward']:
                prop_idx[t][dir] = np.array(prop_idx[t][dir])
                dist = np.abs(prop_idx[t][dir] - t)
                dist = [x if (x <= T / 2) else (T - x) for x in dist]
                sort_idx = np.argsort(dist)
                prop_idx[t][dir] = prop_idx[t][dir][sort_idx]

        # For each time frame, infer the segmentation from its closest annotated time frame
        for t in t_anno:
            for dir in ['forward', 'backward']:
                for target_t in prop_idx[t][dir]:
                    # Propagate from source_t to target_t
                    # To avoid accummulation of sub-pixel errors, use long-range propagation after every 5 frames
                    if np.abs(target_t - t) % 5 == 0:
                        source_t = target_t - 5 if dir == 'forward' else target_t + 5
                    else:
                        source_t = target_t - 1 if dir == 'forward' else target_t + 1
                    source_t = wrap_frame_index([source_t], T)[0]

                    # Perform label propagation
                    print('{0} -> {1}'.format(source_t, target_t))
                    target_image = '{0}/ao_crop_fr{1:02d}.nii.gz'.format(motion_dir, target_t)
                    source_image = '{0}/ao_crop_fr{1:02d}.nii.gz'.format(motion_dir, source_t)
                    par = '{0}/ffd_aortic_motion.cfg'.format(par_path)
                    dof = '{0}/ffd_{1:02d}_to_{2:02d}.dof.gz'.format(motion_dir, target_t, source_t)
                    os.system('mirtk register {0} {1} -parin {2} -dofout {3}'.format(
                        target_image, source_image, par, dof))

                    source_label = '{0}/label_ao_prop{2:02d}.nii.gz'.format(motion_dir, fr, source_t)
                    target_label = '{0}/label_ao_prop{2:02d}.nii.gz'.format(motion_dir, fr, target_t)

                    orig_source_image = '{0}/ao_fr{1:02d}.nii.gz'.format(motion_dir, source_t)
                    os.system('mirtk transform-image {0} {1} -dofin {2} -target {3} -interp NN'.format(
                        source_label, target_label, dof, orig_source_image))

        # Combine into a sequence
        image_names = []
        for t in range(T):
            image_name = '{0}/label_ao_prop{1:02d}.nii.gz'.format(motion_dir, t)
            image_names += [image_name]
        output_name = '{0}/label_ao_prop.nii.gz'.format(data_dir)
        make_sequence(image_names, dt, output_name)

        # Remove intermediate files
        os.system('rm -rf {0}'.format(motion_dir))
