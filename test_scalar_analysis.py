#! /usr/bin/env python

import unittest
import numpy as np
import numpy.testing as npt
import os
import pydicom as dicom
import logging

from phantom_analysis import scalar_analysis, dicom_util

class TestScalarAnalysis(unittest.TestCase):

    def test_t1_calculation_shape(self):
        tr = 1
        alphas = np.array(range(1,6))
        dwi = np.zeros((1,2,3,5))
        t1 = scalar_analysis.calculate_t1(dwi, alphas, tr)

        expected = np.zeros((1,2,3))
        npt.assert_array_equal(expected, t1)

    def test_t1_calculation(self):
        tr = 3.7220 # repetition time taken from sample t1 weighted dicom files

        # test with 1.1 log weight
        alphas = np.array([np.deg2rad(flip_angle_degrees) for flip_angle_degrees in [43,22,15,8,5]])
        dwi = np.array([[[[155, 290, 405, 600, 800]]]])

        t1 = scalar_analysis.calculate_t1(dwi, alphas, tr)
        npt.assert_approx_equal(t1[0][0][0], 1572.275, significant=6)

    # TODO: could still use a better test here
    def test_t1_calculation_slice(self):
        demo_file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/breast_single_t1_slice/")
        dcm_dir = dicom_util.read_dicomdir(demo_file_dir)
        t1_array = scalar_analysis.calculate_t1(dcm_dir["dwi"], dcm_dir["alphas"], dcm_dir["rep_time_seconds"])

        expected_t1 = np.fromfile(os.path.join(demo_file_dir, 't1_test_slice.txt'), sep=' ').reshape(1, 256, 256)
        npt.assert_allclose(t1_array, expected_t1, rtol=1e-07, atol=1e-07)

    def test_adc_calculation_shape(self):
        # check that output arrays are expected shape and size
        dwi = np.zeros((1,3,4,2))
        b_vals = np.array([0, 500])
        adc = scalar_analysis.calculate_adc(dwi, b_vals)

        expected = np.zeros((1,3,4))
        expected *= -1
        npt.assert_array_equal(expected, adc)

    def test_adc_calculation(self):
        # check that ADC calculation works as expected on simple case
        dwi = np.zeros((1,1,1,2))
        dwi[0, 0, 0, 0] = np.exp(-200)
        dwi[0, 0, 0, 1] = np.exp(-500)
        b_vals = np.array([200, 500])
        adc = scalar_analysis.calculate_adc(dwi, b_vals)

        expected = np.zeros((1,1,1))
        expected[0, 0, 0] = 1
        npt.assert_almost_equal(expected, adc)

    # TODO: this test dataset has a different image orientation than our breast phantom cases
    #       so we can't use dicom_util to create the dwi
    def test_hpd_adc_demo(self):
        # test data from HPD
        demo_file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/diffusion_single_adc_slice/")
        file_paths = [os.path.join(demo_file_dir, file) for file in os.listdir(demo_file_dir) if file.endswith('.dcm')]

        # need to read in the DICOM files and get dwi array and bvalue array
        dimages = []
        bvalues = []
        bvalue = (0x0019, 0x100c)
        for file_path in file_paths:
            dcm = dicom.read_file(file_path)
            bvalues.append(int(dcm[bvalue].value))
            dimages.append(dcm.pixel_array)
        self.assertTrue(len(dimages) >= 4)

        bvalues = np.array(bvalues)
        dwi_array = np.array(dimages)
        dwi_array = np.array([np.transpose(dwi_array)])

        calculated_adc = scalar_analysis.calculate_adc(dwi_array, bvalues)
        calculated_adc = np.transpose(calculated_adc[0])
        expected_adc = np.fromfile(os.path.join(demo_file_dir, 'matlab_adc_slice5.txt'), sep=' ').reshape(256, 224)

        # test a value that doesn't rely on details like filtering nor zeroing nans/negatives:
        npt.assert_allclose(calculated_adc[100][100], expected_adc[100][100], rtol=1e-07, atol=1e-07)

        # test all, which hopefully covers all details:
        npt.assert_allclose(calculated_adc, expected_adc, rtol=1e-07, atol=1e-07)

    def test_voi_stats(self):
        # test that it's getting the right number of VOI labels
        # create label_map
        label_map = np.array([[[0, 0, 0], [0, 1, 1], [0, 2, 0]],
                              [[0, 1, 1], [0, 1, 0], [0, 2, 0]],
                              [[0, 0, 0], [0, 0, 0], [0, 2, 0]],
                              [[3, 3, 3], [0, 0, 0], [0, 0, 0]]])
        # create scalar_map, image_coordinate_system, and phantom def
        scalar_map = np.zeros((4,3,3))
        cs = dicom_util.ImageCoordinateSystem(0.15625, 0.2, [-1.0, -0.0, 0.0, -0.0, -1.0, 0.0], 19.9219, 27.7344, -4.3, -4.3)
        phantom_def = { "config": {}, "vois": {
            1: { "expected_value": None, "content_type": "Unknown" },
            2: { "expected_value": None, "content_type": "Unknown" },
            3: { "expected_value": None, "content_type": "Unknown" }
            }}

        voi_stats = scalar_analysis.voi_stats(label_map, scalar_map, cs, phantom_def)
        expected_labels = np.unique(label_map)[1:]
        npt.assert_array_equal(voi_stats.keys(), expected_labels)

        # test that the mean is correct for a single VOI 
        label_map = np.array([[[0, 1, 0]]])
        scalar_map = np.array([[[0, 1572.275, 0]]])
        voi_stats = scalar_analysis.voi_stats(label_map, scalar_map, cs, phantom_def)
        npt.assert_approx_equal(voi_stats[1]["mean"], 1572.275, significant=6)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()