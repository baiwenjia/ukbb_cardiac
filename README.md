## Overview

**ukbb_cardiac** is a toolbox used for processing and analysing cardiovascular magnetic resonance (CMR) images from the [UK Biobank Imaging Study](https://www.ukbiobank.ac.uk/about-our-data/types-of-data/imaging-data/). It consists of several parts:

* pre-processing the original DICOM images, converting them into NIfTI format, which is more convenient for image analysis;
* training fully convolutional networks for short-axis, long-axis and aortic CMR image segmentation;
* deploying the networks to segment images;
* evaluating cardiac imaging phenotypes from the segmentations;
* performing phenome-wide association between imaging phenotypes and non-imaging phenotypes.

This repository only contains the code, not the imaging data. To know more about how to access the UK Biobank imaging data, please go to the [UK Biobank Imaging Study](https://www.ukbiobank.ac.uk/about-our-data/types-of-data/imaging-data/) website. Researchers can [apply](http://www.ukbiobank.ac.uk/register-apply/) to use the UK Biobank data resource for health-related research in the public interest.

**19 Dec 2025** Upon the request of UK Biobank, we deleted the original code repository, reviewed the code to make sure that they do not contain anonymised participant electronic IDs (EIDs), and uploaded the new repository. UK Biobank provides an easy-to-use [code audit tool](https://github.com/UK-Biobank/UKB-Git-Audit-Tool) for checking whether your code contains potential EIDs. If you are a registered user with UK Biobank, there is also a code repository training course that you may need to attend on the UK Biobank AMS system.

## Installation

The toolbox is developed using [Python](https://www.python.org) programming language. Python is usually installed by default on Linux and OSX machines but may need to be installed on Windows machines. Regarding the Python version, I use Python 3. But Python 2 may also work, since I have not used any function specific for Python 3.

The toolbox depends on some external libraries which need to be installed, including:

* tensorflow for deep learning;
* numpy and scipy for numerical computation;
* matplotlib, seaborn for data visulisation;
* pandas and python-dateutil for handling spreadsheet;
* pydicom, SimpleITK for handling dicom images
* nibabel for reading and writing nifti images;
* sciki-image, opencv-python for transforming images in data augmentation;
* vtk for mesh manipulation.
The most convenient way to install these libraries is to use pip3 (or pip for Python 2) by running this command in the terminal:
```
pip3 install tensorflow-gpu numpy scipy matplotlib seaborn pandas python-dateutil pydicom SimpleITK nibabel scikit-image opencv-python vtk
```

The toolbox also evaluates cardiac strain on short-axis and long-axis images. To enable strain calculation, [MIRTK](https://github.com/BioMedIA/MIRTK) needs to be installed. MIRTK is a medical image registration toolbox, which is used for performing cardiac motion tracking on short-axis and long-axis images. However, MIRTK is not a mandatory option for using this toolbox. Without MIRTK, the toolbox will still evaluate most of the cardiac imaging phenotypes, other than strains.

## Usage

**A quick demo** First, please add the github repository directory to your $PYTHONPATH environment, so that the ukbb_cardiac module can be imported and cross-referenced in its code. If you are using Linux, you can run this command:
```
export PYTHONPATH=YOUR_GIT_REPOSITORY_PATH:"${PYTHONPATH}" 
```

To try the demo on the example images, simply run this command:
```
python3 demo_pipeline.py
```
There is one parameter in the script, *CUDA_VISIBLE_DEVICES*, which controls which GPU device to use on your machine. Currently, I set it to 0, which means the first GPU on your machine. This script will download demo cardiac MR images and pre-trained network models, then segment the images and evaluate imaging phenotypes from the segmentation.

**Speed** The speed of image segmentation depends several factors, such as whether to use GPU or CPU, the GPU hardware, the test image size etc. In my case, I use a Nvidia Titan K80 GPU and it takes about 10 seconds to segment a full time sequence (50 time frames), with the image size to be 192x208x10x50 (i.e. each 2D image slice to be 192x208 pixels, with 10 image slices and 50 time frames). Adding the time for short-axis image segmentation, long-axis image segmentation, aortic image segmentation together, it will take about 25 seconds per subject.

For strain evaluation, it will take several minutes per subject, since MIRTK runs on CPUs, instead of GPUs. 

**Data preparation** You will notice there is a directory named *data*, which contains the scripts for preparing the training dataset. For a machine learning project, data preparation step including acquisition, cleaning, format conversion etc normally takes at least the same amount of your time and headache, if nor more, as the machine learning step. But this is a crucial part, as all the following work (your novel machine learning ideas) needs the data.

In my project, I use imaging data from the UK Biobank Imaging Study, which aims to conduct detailed MRI imaging scans of the vital organs of 100,000 participants. Researchers can [apply](http://www.ukbiobank.ac.uk/register-apply/) to use the UK Biobank data resource for health-related research in the public interest.

I have written the following scripts for preparing the UK Biobank cardiac imaging data:
* convert_data_ukbb2964.py and prepare_data_ukbb2964.py, which prepare the cardiac images and manual annotation of 5,000 subjects under UK Biobank Application 2964. This is the dataset that I used for training the network.
* download_data_ukbb_general.py, which shows how to download cardiac MR images and convert the image format from dicom to nifti for a general UK Biobank application. You may adapt this script to prepare your data, if they also come from the UK Biobank.

## References

We would like to thank all the UK Biobank participants and staff who make the CMR imaging dataset possible and also researchers from Queen Mary's University London and Oxford University who performed the hard work of manual annotation. In case you find the toolbox or a certain part of it useful, please consider giving appropriate credit to it by citing one or some of the papers here, which respectively describes the image segmentation methods [1, 2], the full analysis pipeline and phenome-wide association studies [3] and the manual annotation dataset [4]. Thanks.

[1] W. Bai, et al. Automated cardiovascular magnetic resonance image analysis with fully convolutional networks. Journal of Cardiovascular Magnetic Resonance, 20:65, 2018. [doi](https://doi.org/10.1186/s12968-018-0471-x)

[2] W. Bai, et al. Recurrent neural networks for aortic image sequence segmentation with sparse annotations. Medical Image Computing and Computer Assisted Intervention (MICCAI), 2018. [doi](https://doi.org/10.1007/978-3-030-00937-3_67) 

[3] W. Bai, et al. A population-based phenome-wide association study of cardiac and aortic structure and function. Nature Medicine, 2020. [doi](https://www.nature.com/articles/s41591-020-1009-y)

[4] S. Petersen, et al. Reference ranges for cardiac structure and function using cardiovascular magnetic resonance (CMR) in Caucasians from the UK Biobank population cohort. Journal of Cardiovascular Magnetic Resonance, 19:18, 2017. [doi](https://doi.org/10.1186/s12968-017-0327-9)
