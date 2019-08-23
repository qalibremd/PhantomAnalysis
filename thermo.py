#! /usr/bin/env python

import os
import sys
import logging

from phantom_analysis import dicom_util, thermometry

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    if len(sys.argv) == 2:
        demo_dir = sys.argv[1]
    else:
        demo_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/4493056_4493_T1_NIST_PHANTOM_NoPE/E7217/7/")
    dcm_dir = dicom_util.read_dicomdir(demo_dir)

    cs = dcm_dir["image_coordinate_system"]
    temp = thermometry.get_temperature(dcm_dir["dwi"], cs, True)
    logging.info("Temperature: {}".format(temp))
