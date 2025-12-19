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
    The data converting script for UK Biobank Application 2964, which contributes
    the manual annotations of 5,000 subjects.

    This script assumes that the images and annotations have already been downloaded
    as zip files. It decompresses the zip files, sort the DICOM files into subdirectories
    according to the information provided in the manifest.csv spreadsheet, parse manual
    annotated contours from the cvi42 xml files, read the matching DICOM and cvi42 contours
    and finally save them as nifti images.
    """
import os
import csv
import glob
import re
import time
import pandas as pd
import dateutil.parser
from biobank_utils import *
import parse_cvi42_xml


if __name__ == '__main__':
    # Path to the downloaded data
    data_path = 'data_path'

    # For each subdirectory
    for sub_path in sorted(os.listdir(data_path)):
        sub_path = os.path.join(data_path, sub_path)
        # For each subject in the subdirectory
        for eid in sorted(os.listdir(sub_path)):
            data_dir = os.path.join(sub_path, eid)
            # Only convert data if there is manual annotation, i.e. cvi42 files
            if os.path.exists(os.path.join(data_dir, '{0}_cvi42.zip'.format(eid))):
                # Check the annotator's name
                s = os.popen('unzip -c {0}/{1}_cvi42.zip "*.cvi42wsx" '
                             '| grep OwnerUserName'.format(data_dir, eid)).read()
                annotator = (s.split('>')[1]).split('<')[0]

                # Decompress the zip files in this directory
                files = glob.glob('{0}/{1}_*.zip'.format(data_dir, eid))
                dicom_dir = os.path.join(data_dir, 'dicom')
                if not os.path.exists(dicom_dir):
                    os.mkdir(dicom_dir)

                for f in files:
                    if os.path.basename(f) == '{0}_cvi42.zip'.format(eid):
                        os.system('unzip -o {0} -d {1}'.format(f, data_dir))
                    else:
                        os.system('unzip -o {0} -d {1}'.format(f, dicom_dir))

                        # Process the manifest file
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

                # Parse cvi42 xml file
                cvi42_contours_dir = os.path.join(data_dir, 'cvi42_contours')
                if not os.path.exists(cvi42_contours_dir):
                    os.mkdir(cvi42_contours_dir)
                xml_name = os.path.join(data_dir, '{0}_cvi42.cvi42wsx'.format(eid))
                parse_cvi42_xml.parseFile(xml_name, cvi42_contours_dir)

                # Rare cases when no dicom file exists
                if not os.listdir(dicom_dir):
                    print('Warning: empty dicom directory! Skip this one.')
                    continue

                # Convert dicom files and annotations into nifti images
                dset = Biobank_Dataset(dicom_dir, cvi42_contours_dir)
                dset.read_dicom_images()
                dset.convert_dicom_to_nifti(data_dir)

                # Remove intermediate files
                os.system('rm -rf {0} {1}'.format(dicom_dir, cvi42_contours_dir))
                os.system('rm -f {0}'.format(xml_name))
