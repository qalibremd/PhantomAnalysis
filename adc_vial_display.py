#! /usr/bin/env python

import numpy as np
import os
from copy import deepcopy
import logging
import platform

from phantom_analysis import scalar_analysis, dicom_util, voi_analysis, phantom_definitions

WINDOWS = True if platform.system() == 'Windows' else False
CLAMP = (0, 4000)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    demo_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../phantom_data/4493056_4493_T1_NIST_PHANTOM_NoPE/E7217/7")
    #demo_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/voi/adc/breast_131_spheres_adc_20190124_19")
    dcm_dir = dicom_util.read_dicomdir(demo_dir)

    cs = dcm_dir["image_coordinate_system"]
    logging.info("generating vois")
    coronal_dwi = np.transpose(deepcopy(dcm_dir["dwi"]), (3, 1, 0, 2))

    voi_list = voi_analysis.get_vois(dcm_dir["dwi"], cs, phantom_definitions.BREAST_131_CYLS_T1)

    #for v in voi_list["found_vois"]:
        #print("voi {} has height of {} cm".format(v, voi_list["found_vois"][v]["height"]))

    scalar_map = scalar_analysis.calculate_t1(dcm_dir["dwi"], dcm_dir["alphas"], dcm_dir["rep_time_seconds"], use_pool= not WINDOWS, clamp=CLAMP, threshold=5)
    #scalar_map = scalar_analysis.calculate_adc(dcm_dir["dwi"], dcm_dir["bvalues"]) * 1e6
    stats = scalar_analysis.voi_stats(voi_list["label_map"], scalar_map, cs, phantom_definitions.BREAST_131_CYLS_T1 ,clamp=CLAMP)

    for v in stats:
        #print("voi: {}, num clamped pixels: {}".format(v, stats[v]["clamped_pixels"]))
        #print("voi: {}, mean scalar value: {}, expected value: {}".format(v, stats[v]["mean"], phantom_definitions.NEW_ADC["vois"][v]["expected_value"]))
        print(stats[v])