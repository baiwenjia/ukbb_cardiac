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
import pandas as pd
from ukbb_cardiac.common.cardiac_utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', metavar='dir_name', default='', required=True)
    parser.add_argument('--output_csv', metavar='csv_name', default='', required=True)
    parser.add_argument('--output_max_csv', metavar='max_csv_name', default='', required=True)
    args = parser.parse_args()

    data_path = args.data_dir
    data_list = sorted(os.listdir(data_path))
    table = []
    processed_list = []
    table_max = []
    processed_list_max = []
    for data in data_list:
        print(data)
        data_dir = os.path.join(data_path, data)

        # Quality control for segmentation at ED
        # If the segmentation quality is low, evaluation of wall thickness may fail.
        seg_sa_name = '{0}/seg_sa_ED.nii.gz'.format(data_dir)
        if not os.path.exists(seg_sa_name):
            continue
        if not sa_pass_quality_control(seg_sa_name):
            continue

        # Evaluate myocardial wall thickness
        evaluate_wall_thickness('{0}/seg_sa_ED.nii.gz'.format(data_dir),
                                '{0}/wall_thickness_ED'.format(data_dir))

        # Record data
        if os.path.exists('{0}/wall_thickness_ED.csv'.format(data_dir)):
            df = pd.read_csv('{0}/wall_thickness_ED.csv'.format(data_dir), index_col=0)
            line = df['Thickness'].values
            table += [line]
            processed_list += [data]

        if os.path.exists('{0}/wall_thickness_ED_max.csv'.format(data_dir)):
            df = pd.read_csv('{0}/wall_thickness_ED_max.csv'.format(data_dir), index_col=0)
            line = df['Thickness_Max'].values
            table_max += [line]
            processed_list_max += [data]

    # Save wall thickness for all the subjects
    df = pd.DataFrame(table, index=processed_list,
                      columns=['WT_AHA_1 (mm)', 'WT_AHA_2 (mm)', 'WT_AHA_3 (mm)',
                               'WT_AHA_4 (mm)', 'WT_AHA_5 (mm)', 'WT_AHA_6 (mm)',
                               'WT_AHA_7 (mm)', 'WT_AHA_8 (mm)', 'WT_AHA_9 (mm)',
                               'WT_AHA_10 (mm)', 'WT_AHA_11 (mm)', 'WT_AHA_12 (mm)',
                               'WT_AHA_13 (mm)', 'WT_AHA_14 (mm)', 'WT_AHA_15 (mm)', 'WT_AHA_16 (mm)',
                               'WT_Global (mm)'])
    # df.to_csv(args.output_csv)

    df = pd.DataFrame(table_max, index=processed_list_max,
                      columns=['WT_Max_AHA_1 (mm)', 'WT_Max_AHA_2 (mm)', 'WT_Max_AHA_3 (mm)',
                               'WT_Max_AHA_4 (mm)', 'WT_Max_AHA_5 (mm)', 'WT_Max_AHA_6 (mm)',
                               'WT_Max_AHA_7 (mm)', 'WT_Max_AHA_8 (mm)', 'WT_Max_AHA_9 (mm)',
                               'WT_Max_AHA_10 (mm)', 'WT_Max_AHA_11 (mm)', 'WT_Max_AHA_12 (mm)',
                               'WT_Max_AHA_13 (mm)', 'WT_Max_AHA_14 (mm)', 'WT_Max_AHA_15 (mm)', 'WT_Max_AHA_16 (mm)',
                               'WT_Max_Global (mm)'])
    # df.to_csv(args.output_max_csv)
