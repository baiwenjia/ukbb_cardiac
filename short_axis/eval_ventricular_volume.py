# Copyright 2019, Wenjia Bai. All Rights Reserved.
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
# ============================================================================
import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', metavar='dir_name', default='', required=True)
    parser.add_argument('--output_csv', metavar='csv_name', default='', required=True)
    args = parser.parse_args()

    data_path = args.data_dir
    data_list = sorted(os.listdir(data_path))
    table = []
    processed_list = []
    for data in data_list:
        data_dir = os.path.join(data_path, data)
        image_name = '{0}/sa.nii.gz'.format(data_dir)
        seg_name = '{0}/seg_sa.nii.gz'.format(data_dir)

        if os.path.exists(image_name) and os.path.exists(seg_name):
            print(data)

            # Image
            nim = nib.load(image_name)
            pixdim = nim.header['pixdim'][1:4]
            volume_per_pix = pixdim[0] * pixdim[1] * pixdim[2] * 1e-3
            density = 1.05

            # Heart rate
            duration_per_cycle = nim.header['dim'][4] * nim.header['pixdim'][4]
            heart_rate = 60.0 / duration_per_cycle

            # Segmentation
            seg = nib.load(seg_name).get_data()

            frame = {}
            frame['ED'] = 0
            vol_t = np.sum(seg == 1, axis=(0, 1, 2)) * volume_per_pix
            frame['ES'] = np.argmin(vol_t)

            val = {}
            for fr_name, fr in frame.items():
                # Clinical measures
                val['LV{0}V'.format(fr_name)] = np.sum(seg[:, :, :, fr] == 1) * volume_per_pix
                val['LV{0}M'.format(fr_name)] = np.sum(seg[:, :, :, fr] == 2) * volume_per_pix * density
                val['RV{0}V'.format(fr_name)] = np.sum(seg[:, :, :, fr] == 3) * volume_per_pix

            val['LVSV'] = val['LVEDV'] - val['LVESV']
            val['LVCO'] = val['LVSV'] * heart_rate * 1e-3
            val['LVEF'] = val['LVSV'] / val['LVEDV'] * 100

            val['RVSV'] = val['RVEDV'] - val['RVESV']
            val['RVCO'] = val['RVSV'] * heart_rate * 1e-3
            val['RVEF'] = val['RVSV'] / val['RVEDV'] * 100

            line = [val['LVEDV'], val['LVESV'], val['LVSV'], val['LVEF'], val['LVCO'], val['LVEDM'],
                    val['RVEDV'], val['RVESV'], val['RVSV'], val['RVEF']]
            table += [line]
            processed_list += [data]

    df = pd.DataFrame(table, index=processed_list,
                      columns=['LVEDV (mL)', 'LVESV (mL)', 'LVSV (mL)', 'LVEF (%)', 'LVCO (L/min)', 'LVM (g)',
                               'RVEDV (mL)', 'RVESV (mL)', 'RVSV (mL)', 'RVEF (%)'])
    df.to_csv(args.output_csv)
