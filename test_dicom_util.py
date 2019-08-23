#! /usr/bin/env python

import unittest
import numpy as np
import pydicom as dicom
import numpy.testing as npt
import os
import logging

from phantom_analysis import dicom_util

def create_dcm_dataset():
    file_meta = dicom.dataset.Dataset()
    file_meta.TransferSyntaxUID = "1.2.840.10008.1.2"

    data = [ dicom.dataset.Dataset() for i in range(5) ]
    for i in range(5):
        dcm = data[i]
        dcm.Manufacturer = "SIEMENS"
        dcm.file_meta = file_meta
        dcm.is_little_endian = True
        dcm.is_implicit_VR = True
        dcm.PixelRepresentation = 1
        dcm.BitsAllocated = 16
        dcm.SamplesPerPixel = 1
        dcm.PixelData = np.zeros((1,1)).tostring()
        dcm.InStackPositionNumber = 1
        dcm.Rows = 2
        dcm.Columns = 2

    return data

def create_bvalue_dataset():
    data = create_dcm_dataset()
    for i in range(len(data)):
        dcm = data[i]
        dcm[(0x0019, 0x100c)] = dicom.dataset.DataElement(0x0019100c, 'bvalue', '{}'.format(i*100))
    return data

def create_flip_dataset():
    data = create_dcm_dataset()
    for i in range(len(data)):
        dcm = data[i]
        dcm.FlipAngle = '{}'.format(i*10)
    return data

class TestDicomUtil(unittest.TestCase):

    def test_b_value(self):
        data = dicom.dataset.Dataset()

        # test Siemens
        data.Manufacturer = "SIEMENS"
        data[(0x0019, 0x100c)] = dicom.dataset.DataElement(0x0019100c, 'bvalue', '100')
        b_value = dicom_util.get_b_value(data)
        self.assertEqual(b_value, 100)

        # test philips
        data.Manufacturer = "PHILIPS HEALTHCARE"
        data[(0x0018, 0x9087)] = dicom.dataset.DataElement(0x00189087, 'bvalue', '800')
        b_value = dicom_util.get_b_value(data)
        self.assertEqual(b_value, 800)

        # test unknown manufacturer
        data.Manufacturer = "unknown provider"
        self.assertRaises(Exception, dicom_util.get_b_value, data)

        # test GE (multiple)
        data.Manufacturer = "GE MEDICAL SYSTEMS"
        data[(0x0043, 0x1039)] = dicom.dataset.DataElement(0x00431039, 'OB', '1000\\8\\0\\0')
        b_value = dicom_util.get_b_value(data)
        self.assertEqual(b_value, 1000)

        data[(0x0043, 0x1039)] = dicom.dataset.DataElement(0x00431039, 'IS', ['1000000900', '8', '0', '0'])
        b_value = dicom_util.get_b_value(data)
        self.assertEqual(b_value, 900)

    def test_get_all_bvalues(self):
        data = create_bvalue_dataset()
        bvalues = dicom_util.get_all_bvalues(data)
        expected = np.array([0, 100, 200, 300, 400])
        npt.assert_array_equal(bvalues, expected)

    def test_get_all_flipangles(self):
        data = create_flip_dataset()
        flips = dicom_util.get_all_flipangles(data)
        expected = np.array([0, 10, 20, 30, 40])
        npt.assert_array_equal(flips, expected)

    def test_get_repetition_time(self):
        data = dicom.dataset.Dataset()
        data[(0x0018, 0x0080)] = dicom.dataset.DataElement(0x00180080, 'repetition time', '10')
        tr = dicom_util.repetition_time(data)
        self.assertEqual(tr, 10)

    def test_get_dwi_array(self):
        data_bvalues = create_bvalue_dataset()
        data_flips = create_flip_dataset()
        # test for invalid second parameter
        self.assertRaises(Exception, dicom_util.get_dwi_array, data_bvalues, "invalid parameter")

        # test with all zeros, 5 bvalues/flip angles
        expected = np.zeros((1,2,2,5))
        dwi_bvalues = dicom_util.get_dwi_array(data_bvalues, "BVALUE")
        dwi_flips = dicom_util.get_dwi_array(data_flips, "FLIP")
        npt.assert_array_equal(dwi_bvalues, expected)
        npt.assert_array_equal(dwi_flips, expected)

        # test bvalues with invalid parameters
        data_bvalues.append(dicom.dataset.Dataset())
        dcm = data_bvalues[5]
        dcm.Manufacturer = "SIEMENS"
        dcm.file_meta = data_bvalues[0].file_meta
        dcm.is_little_endian = True
        dcm.is_implicit_VR = True
        dcm.PixelRepresentation = 1
        dcm.BitsAllocated = 16
        dcm.SamplesPerPixel = 1
        dcm.PixelData = np.zeros((1,1)).tostring()
        dcm.InStackPositionNumber = 2
        dcm.Rows = 2
        dcm.Columns = 2
        dcm[(0x0019, 0x100c)] = dicom.dataset.DataElement(0x0019100c, 'bvalue', '{}'.format(600))
        self.assertRaises(Exception, dicom_util.get_dwi_array, data_bvalues, "BVALUE")

        # test bvalues with different pixels for averaging
        dcm = data_bvalues[5] 
        dcm.InStackPositionNumber = 1
        dcm[(0x0019, 0x100c)] = dicom.dataset.DataElement(0x0019100c, 'bvalue', '{}'.format(0))
        for i in range(len(data_bvalues)):
            dcm = data_bvalues[i]
            pixels = dcm.pixel_array
            pixels += 1
            pixels *= dicom_util.get_b_value(dcm)
            dcm.PixelData = pixels.tobytes()

        dwi_bvalues = dicom_util.get_dwi_array(data_bvalues, "BVALUE")
        expected = np.array([[[[0,100,200,300,400],[0,100,200,300,400]],[[0,100,200,300,400],[0,100,200,300,400]]]])
        npt.assert_array_equal(dwi_bvalues, expected)

        dcm = data_bvalues[5]
        pixels =dcm.pixel_array
        pixels += 100
        dcm.PixelData = pixels.tobytes()

        dwi_bvalues = dicom_util.get_dwi_array(data_bvalues, "BVALUE")
        expected = np.array([[[[50,100,200,300,400],[50,100,200,300,400]],[[50,100,200,300,400],[50,100,200,300,400]]]])
        npt.assert_array_equal(dwi_bvalues, expected)

        # test flip angles with invalid parameters
        data_flips.append(dicom.dataset.Dataset())
        dcm = data_flips[5]
        dcm.InStackPositionNumber = 2
        dcm.FlipAngle = 60
        self.assertRaises(Exception, dicom_util.get_dwi_array, data_flips, "FLIP")

    # test image coordinate system
    def test_image_coordinate_system(self):
        # check coordinate system creation with invalid image orientation
        self.assertRaises(Exception, dicom_util.ImageCoordinateSystem, 0.15625, 0.2, [1.0, -0.0, 0.0, -0.0, -1.0, 0.0], 19.9219, 27.7344, -4.3, -4.3)

        # test expected behavior for this dataset
        expected = dicom_util.ImageCoordinateSystem(0.15625, 0.2, [-1.0, -0.0, 0.0, -0.0, -1.0, 0.0], 19.9219, 27.7344, -4.3, -4.3)
        self.assertListEqual(expected.cs_to_array(), [0.15625, 0.2, 19.9219, 27.7344, -4.3, -4.3, [-1.0, -0.0, 0.0, -0.0, -1.0, 0.0]])
        # test image coordinate system functions
        self.assertEqual(expected.lps_cm(0,0,0), (19.9219, 27.7344, -4.3))
        self.assertEqual(expected.coronal_center_to_pixels(0.0, 0.0), (128, -21))
        self.assertEqual(expected.coronal_z_to_cm(10), 26.1719)
        self.assertListEqual(expected.coronal_circle_to_cm(200, 200, 1.0), [-11.3281, -44.3, 0.2])

    # test splitting on bvalues vs flip angles and compare to known datasets
    def test_dicom_dir(self):
        # ADC case
        demo_file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/breast_single_adc_slice/")
        dir_output = dicom_util.read_dicomdir(demo_file_dir)
        expected_cs = dicom_util.ImageCoordinateSystem(0.15625, 0.2, [-1.0, -0.0, 0.0, -0.0, -1.0, 0.0], 19.9219, 27.7344, -4.3, -4.3)

        self.assertEqual(dir_output["scalar_type"], "ADC")
        npt.assert_array_equal(dir_output["bvalues"], [0, 600])
        self.assertEqual(np.shape(dir_output["dwi"]), (1,256,256,2))
        self.assertListEqual(dir_output["image_coordinate_system"].cs_to_array(), expected_cs.cs_to_array())

        # T1 case
        demo_file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/breast_single_t1_slice/")
        dir_output = dicom_util.read_dicomdir(demo_file_dir)
        expected_cs = dicom_util.ImageCoordinateSystem(0.15625, 0.2, [-1.0, -0.0, 0.0, -0.0, -1.0, 0.0], 20.0019, 20.6719, -0.16999999999999998, -0.16999999999999998)

        self.assertEqual(dir_output["scalar_type"], "T1")
        self.assertListEqual(dir_output["image_coordinate_system"].cs_to_array(), expected_cs.cs_to_array())
        self.assertEqual(np.shape(dir_output["dwi"]), (1,256,256,5))
        npt.assert_array_equal(dir_output["flip_angles_degrees"], [5, 8, 15, 22, 43])
        self.assertEqual(dir_output["rep_time_seconds"], 3)
        npt.assert_allclose(dir_output["alphas"], np.array([0.08726646, 0.13962634, 0.26179939, 0.38397244, 0.75049158]))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()