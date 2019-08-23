#! /usr/bin/env python

import os
import logging
import numpy as np

from phantom_analysis import dicom_util, voi_analysis, phantom_definitions


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)


    FILES = [ "../data/4493056_4493_T1_NIST_PHANTOM_NoPE/E7217/10", "../data/4493056_4493_T1_NIST_PHANTOM_NoPE/E7217/7", "test_data/voi/t1/20190124_7"]
    DEFS = [ phantom_definitions.BREAST_131_REVISIONB_ADC, phantom_definitions.BREAST_131_REVISIONB_T1, phantom_definitions.BREAST_131_VERSION3_T1]

    # For each case
    for filename,phantom_def in zip(FILES, DEFS):
		logging.info("file: {}".format(filename))
		demo_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
		dcm_dir = dicom_util.read_dicomdir(demo_dir)
		voi_dir = voi_analysis.get_vois(dcm_dir["dwi"], dcm_dir["image_coordinate_system"], phantom_def)
		voi_list = voi_dir["found_vois"]

		i = 0
		for v in voi_list.keys():
			max_radius = phantom_def["vois"][v]["radius_cm"] * phantom_def["config"]["radius_max_percentage"]
			found_radius = voi_list[v]["radius"]
			if found_radius > max_radius and np.abs(found_radius - max_radius) > 1e-6:
				i = i + 1
				logging.warn("max radius is {} and found radius for voi {} is {}".format(max_radius, v, found_radius))

		if i > 0:
			logging.info("FAIL -- {} cases where radius is greater than max".format(i))
		else:
			logging.info("PASS")