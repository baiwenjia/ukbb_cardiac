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
    This script demonstrates a pipeline for cardiac MR image analysis.
    """
import os
import urllib.request
import shutil


if __name__ == '__main__':
    # The GPU device id
    CUDA_VISIBLE_DEVICES = 0

    # The URL for downloading demo data
    URL = 'https://www.doc.ic.ac.uk/~wbai/data/ukbb_cardiac/'

    # Download demo images
    print('Downloading demo images ...')
    for i in [1, 2]:
        if not os.path.exists('demo_image/{0}'.format(i)):
            os.makedirs('demo_image/{0}'.format(i))
        for seq_name in ['sa', 'la_2ch', 'la_4ch', 'ao']:
            f = 'demo_image/{0}/{1}.nii.gz'.format(i, seq_name)
            urllib.request.urlretrieve(URL + f, f)

    # Download information spreadsheet
    print('Downloading information spreadsheet ...')
    if not os.path.exists('demo_csv'):
        os.makedirs('demo_csv')
    for f in ['demo_csv/blood_pressure_info.csv']:
        urllib.request.urlretrieve(URL + f, f)

    # Download trained models
    print('Downloading trained models ...')
    if not os.path.exists('trained_model'):
        os.makedirs('trained_model')
    for model_name in ['FCN_sa', 'FCN_la_2ch', 'FCN_la_4ch', 'FCN_la_4ch_seg4', 'UNet-LSTM_ao']:
        for f in ['trained_model/{0}.meta'.format(model_name),
                  'trained_model/{0}.index'.format(model_name),
                  'trained_model/{0}.data-00000-of-00001'.format(model_name)]:
            urllib.request.urlretrieve(URL + f, f)

    # Analyse show-axis images
    print('******************************')
    print('  Short-axis image analysis')
    print('******************************')

    # Deploy the segmentation network
    print('Deploying the segmentation network ...')
    os.system('CUDA_VISIBLE_DEVICES={0} python3 common/deploy_network.py --seq_name sa --data_dir demo_image '
              '--model_path trained_model/FCN_sa'.format(CUDA_VISIBLE_DEVICES))

    # Evaluate ventricular volumes
    print('Evaluating ventricular volumes ...')
    os.system('python3 short_axis/eval_ventricular_volume.py --data_dir demo_image '
              '--output_csv demo_csv/table_ventricular_volume.csv')

    # Evaluate wall thickness
    print('Evaluating myocardial wall thickness ...')
    os.system('python3 short_axis/eval_wall_thickness.py --data_dir demo_image '
              '--output_csv demo_csv/table_wall_thickness.csv')

    # Evaluate strain values
    if shutil.which('mirtk'):
        print('Evaluating strain from short-axis images ...')
        os.system('python3 short_axis/eval_strain_sax.py --data_dir demo_image '
                  '--par_dir par --output_csv demo_csv/table_strain_sax.csv')

    # Analyse long-axis images
    print('******************************')
    print('  Long-axis image analysis')
    print('******************************')

    # Deploy the segmentation network
    print('Deploying the segmentation network ...')
    os.system('CUDA_VISIBLE_DEVICES={0} python3 common/deploy_network.py --seq_name la_2ch --data_dir demo_image '
              '--model_path trained_model/FCN_la_2ch'.format(CUDA_VISIBLE_DEVICES))

    os.system('CUDA_VISIBLE_DEVICES={0} python3 common/deploy_network.py --seq_name la_4ch --data_dir demo_image '
              '--model_path trained_model/FCN_la_4ch'.format(CUDA_VISIBLE_DEVICES))

    os.system('CUDA_VISIBLE_DEVICES={0} python3 common/deploy_network.py --seq_name la_4ch --data_dir demo_image '
              '--seg4 --model_path trained_model/FCN_la_4ch_seg4'.format(CUDA_VISIBLE_DEVICES))

    # Evaluate atrial volumes
    print('Evaluating atrial volumes ...')
    os.system('python3 long_axis/eval_atrial_volume.py --data_dir demo_image '
              '--output_csv demo_csv/table_atrial_volume.csv')

    # Evaluate strain values
    if shutil.which('mirtk'):
        print('Evaluating strain from long-axis images ...')
        os.system('python3 long_axis/eval_strain_lax.py --data_dir demo_image '
                  '--par_dir par --output_csv demo_csv/table_strain_lax.csv')

    # Analyse aortic images
    print('******************************')
    print('  Aortic image analysis')
    print('******************************')

    # Deploy the segmentation network
    print('Deploying the segmentation network ...')
    os.system('CUDA_VISIBLE_DEVICES={0} python3 common/deploy_network_ao.py --seq_name ao --data_dir demo_image '
              '--model_path trained_model/UNet-LSTM_ao'.format(CUDA_VISIBLE_DEVICES))

    # Evaluate aortic areas
    print('Evaluating atrial areas ...')
    os.system('python3 aortic/eval_aortic_area.py --data_dir demo_image '
              '--pressure_csv demo_csv/blood_pressure_info.csv --output_csv demo_csv/table_aortic_area.csv')

    print('Done.')
