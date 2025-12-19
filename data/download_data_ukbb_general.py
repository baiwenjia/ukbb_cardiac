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
    The script downloads the cardiac MR images for a UK Biobank Application and
    converts the DICOM into nifti images.
    """
import os
import glob
import pandas as pd
from biobank_utils import *
import dateutil.parser


if __name__ == '__main__':
    # Where the data will be downloaded
    data_root = 'data_path'

    # Path to the UK Biobank utilities directory
    # The utility programmes can be downloaded at http://biobank.ctsu.ox.ac.uk/crystal/download.cgi
    util_dir = 'util_path'

    # The authentication file (application id + password) for downloading the data for a specific
    # UK Biobank application. You will get this file from the UK Biobank website after your
    # data application and data download exemption have been approved.
    ukbkey = 'my_ukbkey'

    # The spreadsheet which lists the anonymised IDs of the subjects.
    # You can download a very large spreadsheet from the UK Biobank website, which exceeds 10GB.
    # I normally first filter the spreadsheet, select only a subset of subjects with imaging data
    # and save them in a smaller spreadsheet.
    csv_file = 'ukb_image_subset.csv'
    df = pd.read_csv(os.path.join(csv_dir, csv_file), header=1)
    data_list = df['eid']

    # Download cardiac MR images for each subject
    start_idx = 0
    end_idx = len(data_list)
    for i in range(start_idx, end_idx):
        eid = str(data_list[i])

        # Destination directories
        data_dir = os.path.join(data_root, eid)
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        dicom_dir = os.path.join(data_dir, 'dicom')
        if not os.path.exists(dicom_dir):
            os.mkdir(dicom_dir)

        # Create a batch file for this subject
        batch_file = os.path.join(data_dir, '{0}_batch'.format(eid))
        with open(batch_file, 'w') as f_batch:
            for j in range(20208, 20210):
                # The field ID information can be searched at http://biobank.ctsu.ox.ac.uk/crystal/search.cgi
                # 20208: Long axis heart images - DICOM Heart MRI
                # 20209: Short axis heart images - DICOM Heart MRI
                # 2.0 means the 2nd visit of the subject, the 0th data item for that visit.
                # The first imaging scan for each subject is performed at his/her 2nd visit.
                # The second image scan, i.e. repeated imaging scan, is performed at their 3rd visit.
                field = '{0}-2.0'.format(j)
                f_batch.write('{0} {1}_2_0\n'.format(eid, j))

        # Download data using the batch file
        ukbfetch = os.path.join(util_dir, 'ukbfetch')
        print('{0}: Downloading data for subject {1} ...'.format(i, eid))
        os.system('{0} -b{1} -a{2}'.format(ukbfetch, batch_file, ukbkey))

        # Unpack the data
        files = glob.glob('{0}_*.zip'.format(eid))
        for f in files:
            os.system('unzip -o {0} -d {1}'.format(f, dicom_dir))

            # Process the manifest file
            if os.path.exists(os.path.join(dicom_dir, 'manifest.cvs')):
                os.system('cp {0} {1}'.format(os.path.join(dicom_dir, 'manifest.cvs'),
                                              os.path.join(dicom_dir, 'manifest.csv')))
            process_manifest(os.path.join(dicom_dir, 'manifest.csv'),
                             os.path.join(dicom_dir, 'manifest2.csv'))
            df2 = pd.read_csv(os.path.join(dicom_dir, 'manifest2.csv'), error_bad_lines=False)

            # Organise the dicom files
            # Group the files into subdirectories for each imaging series
            for series_name, series_df in df2.groupby('series discription'):
                series_dir = os.path.join(dicom_dir, series_name)
                if not os.path.exists(series_dir):
                    os.mkdir(series_dir)
                series_files = [os.path.join(dicom_dir, x) for x in series_df['filename']]
                os.system('mv {0} {1}'.format(' '.join(series_files), series_dir))

        # Convert dicom files and annotations into nifti images
        dset = Biobank_Dataset(dicom_dir)
        dset.read_dicom_images()
        dset.convert_dicom_to_nifti(data_dir)

        # Remove intermediate files
        os.system('rm -rf {0}'.format(dicom_dir))
        os.system('rm -f {0}'.format(batch_file))
        os.system('rm -f {0}_*.zip'.format(eid))
