## Overview

**ukbb_cardiac** is a toolbox used for processing and analysing cardiovascular magnetic resonance (CMR) images from the [UK Biobank Imaging Study](http://imaging.ukbiobank.ac.uk/). It consists of several parts:

* pre-processing the original DICOM images, converting them into NIfTI format, which is more convenient for image analysis;
* training fully convolutional networks for short-axis, long-axis and aortic CMR image segmentation;
* deploying the networks to segment images;
* evaluating cardiac imaging phenotypes from the segmentations;
* performing phenome-wide association between imaging phenotypes and non-imaging phenotypes.

**Note** Upon the request of UK Biobank, we have temporarily deleted the code repository. Very sorry about this. We will re-upload the code files, after reviewing and cleaning up to make sure they do not contain anonymised electronic IDs (eIDs) in the code.
