#! /usr/bin/env python

import unittest
import numpy as np
import numpy.testing as npt
import cv2
import logging

from phantom_analysis import dicom_util, thermometry

class TestThermometry(unittest.TestCase):

	def test_ref_array(self):
		ref = thermometry.get_ref_array(0.03226)
		exp = np.zeros((240, 372), dtype=np.uint8)

		for y in (29, 76, 170, 217):
			for x in range(8 if y in (76, 170) else 7):
				cv2.circle(exp, (23+x*46, y), 15, 255, 2)

		npt.assert_array_equal(exp, ref)

	def test_bubble_coordinates(self):
		bubble = thermometry.get_bubble_coordinates(0.06445, (0,0), 0)
		expected = (5, 58, 15, 66)
		npt.assert_array_equal(bubble, expected)

	def test_case(self):
		test_dir = "test_data/thermometry"
		try:
			dir_output = dicom_util.read_dicomdir(test_dir)
		except Exception:
			logging.warn("Failed to read DICOM files.  If DICOM files are not present, they can be populated with "
                  "`aws s3 sync s3://hpd-phantom-data/thermometry %s`" % test_dir)
			raise
		temp = thermometry.get_temperature(dir_output["dwi"], dir_output["image_coordinate_system"], False)
		expected_temp = 23
		self.assertEqual(temp, expected_temp)	


if __name__ == "__main__":
	logging.getLogger().setLevel(logging.INFO)
	unittest.main()