# functions for interacting with dicom files
import os
import numpy as np
from collections import defaultdict
import logging
import time
from datetime import datetime

try:
    import pydicom as dicom
    logging.info("using pydicom")
except:
    import dicom
    logging.info("using dicom")


class InvalidDataException(Exception):
    pass


class ImageCoordinateSystem(object):
    """
    Defines image LPS (left, posterior, superior) position from DICOM attributes and a slice's x y pixels.
    """

    def __init__(self, pixel_spacing_cm, spacing_between_slices_cm, image_orientation_patient,
                 pixel_x0_left_cm, pixel_y0_posterior_cm, min_superior_cm, max_superior_cm):
        """
        Args:
            pixel_spacing_cm: float from DICOM "Pixel Spacing" attribute
            spacing_between_slices_cm: float from DICOM "Spacing Between Slices" attribute
            pixel_x0_left_cm: float from DICOM "Image Position (Patient)" attribute
            pixel_y0_posterior_cm: float from DICOM "Image Position (Patient)" attribute
            min_superior_cm: float from DICOM "Image Position (Patient)" attribute
        """
        self.pixel_spacing_cm = pixel_spacing_cm
        self.spacing_between_slices_cm = spacing_between_slices_cm
        self.pixel_x0_left_cm = pixel_x0_left_cm
        self.pixel_y0_posterior_cm = pixel_y0_posterior_cm
        self.min_superior_cm = min_superior_cm
        self.max_superior_cm = max_superior_cm

        negative_axial_image_orientation_patient = [-1.0, -0.0, 0.0, -0.0, -1.0, 0.0]
        if image_orientation_patient != negative_axial_image_orientation_patient:
            # we may need to add support for other orientations later, but deferring for now
            raise InvalidDataException("Only negative axial image orientation patient supported: expected %s but got %s" % (
                negative_axial_image_orientation_patient, image_orientation_patient))
        self.image_orientation = image_orientation_patient

    # return an array of coordinate system values for unit testing and printing
    def cs_to_array(self):
        return [
            self.pixel_spacing_cm,
            self.spacing_between_slices_cm,
            self.pixel_x0_left_cm,
            self.pixel_y0_posterior_cm,
            self.min_superior_cm,
            self.max_superior_cm,
            self.image_orientation
        ]

    def lps_cm(self, pixel_x, pixel_y, slice):
        """
        Returns: (left, posterior, superior) position in centimeters relative to DICOM "Image Position (Patient)"

        From DICOM tag "Image Position (Patient)", we get the top left coordinate cm of x=0 and y=0.

        The image has left as low x and right as high x, so we subtract the pixel x to get the left coordinate.
        A positive center_left_cm value means "left" and a negative value means "right".

        The image has posterior as low y and anterior as high y, so we subtract the pixel y to get the posterior
        coordinate.  A positive center_posterior_cm value means "anterior" and a negative value means "posterior".

        This is consistent with how slicer and UCSF uses the LPS coordinate system.  We may need to account for the
        "Image Orientation (Patient)" tag if this logic breaks with future data sets.

        e.g. assuming:
            * 256x256 pixel image per slice
            * "Pixel Spacing" is "1.5625 1.5625"
            * "Image Position (Patient)" is 206.919, 151.819, 76
        then:
            * x value 0 (center of first pixel) becomes 206.919 mm or 20.6919 cm
            * x value 255 (center of last pixel) becomes 206.919 - 255*1.5625 = -191.5185 mm = -19.15185 cm
            * x value 128 (center of middle pixel) becomes 206.919 - 128*1.5625 = 6.919 mm = 0.6919 cm
            * y value 0 (center of first pixel) becomes 151.819 mm or 15.1819 cm
            * y value 255 (center of last pixel) becomes 151.819 - 255*1.5625 = -246.6185 mm = -24.66185 cm
            * y value 128 (center of middle pixel) becomes 151.819 - 128*1.5625 = -48.181 mm = -4.8181 cm
        """
        return (
            self.pixel_x0_left_cm - (pixel_x * self.pixel_spacing_cm),       # left
            self.pixel_y0_posterior_cm - (pixel_y * self.pixel_spacing_cm),  # posterior
            self.min_superior_cm + (slice * self.spacing_between_slices_cm)  # superior
        )

    # get coronal center coordinates and radius in cm
    def coronal_circle_to_cm(self, coronal_x, coronal_y, radius_pixels):
        return [
            self.pixel_x0_left_cm - (coronal_x * self.pixel_spacing_cm),           # left
            self.max_superior_cm - (coronal_y * self.spacing_between_slices_cm),   # superior
            radius_pixels * max(self.pixel_spacing_cm, self.spacing_between_slices_cm)
        ]

    # get coronal center in pixels (as ints) for creating label map
    def coronal_center_to_pixels(self, coronal_x_cm, coronal_y_cm):
        return (
            int(np.rint((self.pixel_x0_left_cm - coronal_x_cm) / self.pixel_spacing_cm)),
            int(np.rint((self.max_superior_cm - coronal_y_cm) / self.spacing_between_slices_cm)),
        )

    # get coronal z value in cm
    def coronal_z_to_cm(self, coronal_slice):
        return self.pixel_y0_posterior_cm - (coronal_slice * self.pixel_spacing_cm)


def get_datetime(dcm):
    """ Returns datetime object from a DICOM DA datatype and an optional DICOM TM datatype.

    If TM is None or cannot be parsed, attempts to parse just the DA.
    If fails to parse, logs warning.
    If da cannot be parsed, returns None (doesn't raise exception).

    See http://northstar-www.dartmouth.edu/doc/idl/html_6.2/Value_Representations.html for
    more details
    """
    da = dcm.StudyDate
    tm = dcm.SeriesTime or dcm.StudyTime

    date_format = '%Y%m%d'
    fmts = []
    try:

        fmts = [date_format + ' %H%M%S.%f',
                date_format + ' %H:%M:%S.%f',
                date_format + ' %H%M',
                date_format + ' %H%M%S']

        if tm is not None:
            for fmt in fmts:
                try:
                    return datetime.strptime("%s %s" % (da, tm), fmt)
                except ValueError:
                    pass
        return datetime.strptime(da, date_format)
    except ValueError:
        return None


# get the b_value for a dicom object
def get_b_value(dcm):
    manufacturer = dcm.Manufacturer.upper()
    if manufacturer in ('PHILIPS MEDICAL SYSTEMS', 'PHILIPS HEALTHCARE'):
        key = (0x0018, 0x9087)
    elif manufacturer == 'SIEMENS':
        key = (0x0019, 0x100c)
    elif manufacturer == 'GE MEDICAL SYSTEMS':
        key = (0x0043, 0x1039)
    else:
        raise InvalidDataException("Unknown location of bvalue for manufacturer %s" % manufacturer)

    if manufacturer != 'GE MEDICAL SYSTEMS':
        return int(dcm[key].value)

    # GE can be wacky
    bvalue_data_element = dcm[key]
    # GE bvalue type is either "OB" (Other Byte String) or "IS" (Integer String)
    # for bvalue 1000, an example string is '1000\\8\\0\\0'
    # for bvalue 1000, an example "integer string" is ['1000001000', '8', '0', '0']
    if bvalue_data_element.VR == 'OB':
        bvalue = int(dcm[key].value.split('\\')[0])
    else:
        bvalue = int(dcm[key].value[0])
        if bvalue >= 10 ** 9:
            # e.g. a GE bvalue may be "1000000900"
            # "DW-MRI vendor-specific tags.xlsx" suggests subtracting 10^6, but 10^9 seems to work better
            bvalue -= 10 ** 9
    return bvalue


# get all bvalues for a set of dicom objects
def get_all_bvalues(dicom_array):
    bvalues = set()
    for dcm in dicom_array:
        bvalues.add(get_b_value(dcm))
    return np.array(sorted(list(bvalues)))


# get the repetition time for a dicom object
def repetition_time(dcm):
    key = (0x0018, 0x0080)
    return int(dcm[key].value) if key in dcm else None


# get the spacing between slices for a dicom object
def spacing_between_slices_cm(dcm):
    # SpacingBetweenSlices is in mm as a "Decimal String (DS)"
    return float(dcm.SpacingBetweenSlices)/10


# get the pixel spacing for a dicom object
def pixel_spacing_cm(dcm):
    # PixelSpacing is in mm as a "Decimal String" in row column order
    if len(dcm.PixelSpacing) != 2 or dcm.PixelSpacing[0] != dcm.PixelSpacing[1]:
        # we can probably support unequal row/column spacing, but if this doesn't occur in practice lets not bother
        raise InvalidDataException("Expected equal row and column pixel spacing but got %s" % dcm.PixelSpacing)
    return float(dcm.PixelSpacing[0])/10


# get the patient position for a dicom object
def image_position_patient_cm(dcm):
    return [float(p)/10 for p in dcm.ImagePositionPatient]


# get the image orientation for a dicom object
def image_orientation_patient(dcm):
    return [float(p) for p in dcm.ImageOrientationPatient]


def iop_string(iop):
    return {
        (1,  0,  0,  0,  0, -1): "Coronal",
        (1,  0,  0,  0,  1,  0):  "Axial",
        (-1,  0,  0,  0, -1,  0): "Axial (Negative)",
        (0,  1,  0,  0,  0, -1):  "Sagital"
    }.get(tuple(iop))


# get all the flip angles (in degrees) for a set of dicom objects
def get_all_flipangles(dicom_array):
    flip_angles = set()
    for dcm in dicom_array:
        flip_angles.add(int(dcm.FlipAngle))
    return np.array(sorted(list(flip_angles)))


# get the dwi array for a set of dicoms based on bvalue or flip angle
def get_dwi_array(dicom_array, sorting_param):
    if sorting_param not in ("FLIP", "BVALUE"):
        raise InvalidDataException("Expected a valid parameter (FLIP or BVALUE), got: {}".format(sorting_param))

    # order dicoms by position then bvalue/flip angle
    stack_pos_to_param_to_dcm = defaultdict(lambda: defaultdict(list))
    for dcm in dicom_array:
        stack_pos_to_param_to_dcm[dcm.InStackPositionNumber][dcm.FlipAngle if sorting_param == "FLIP" else get_b_value(dcm)].append(dcm)

    dwi_array = []
    params = None
    for stack_pos in sorted(stack_pos_to_param_to_dcm.keys()):
        dwi_for_stack_pos = []
        if params is None:
            params = sorted(stack_pos_to_param_to_dcm[stack_pos].keys())
        else:
            if params != sorted(stack_pos_to_param_to_dcm[stack_pos].keys()):
                raise InvalidDataException("Inconsistent secondary parameters: expected %s, got %s" % params, sorted(stack_pos_to_param_to_dcm[stack_pos].keys()))
        for param in params:
            pixel_arrays = [dcm.pixel_array for dcm in stack_pos_to_param_to_dcm[stack_pos][param]]
            avg_array = np.mean(pixel_arrays, axis=0, dtype=pixel_arrays[0].dtype)
            dwi_for_stack_pos.append(np.transpose(avg_array))  # transpose will be undone at the end of the for loop

        dwi_array.append(np.transpose(dwi_for_stack_pos))  # this ensures z, x, y, then b/flip indexing order

    dwi_array = np.array(dwi_array)
    return dwi_array


def get_summary_info(dcm):
    return {'study_date': get_datetime(dcm).isoformat(),
            'series': dcm.SeriesNumber,
            'series_description': dcm.SeriesDescription,
            'position': ", ".join([str(10*x) for x in image_position_patient_cm(dcm)]),
            'bvalue': get_b_value(dcm),
            'flip_angle': dcm.FlipAngle}


# takes a directory and returns the information about the dicom dataset
#      what type of dataset: T1, ADC, or thermometry
#      the dwi array for the dataset
#      flip angles (in degrees and radians) or bvalues
#      repetition time for T1 datasets
# TODO: do we want to return bvalues/flip angles and rep time for all cases and some would just have one value? would this be useful for errors or calibration later?
def read_dicomdir(directory):
    before = time.time()
    dicoms = []
    dicom_summaries = {}
    between_slices_cm = None
    pixel_cm = None
    pixel_x0_left_cm = None
    pixel_y0_posterior_cm = None
    orientation = None
    min_superior_cm, max_superior_cm = None, None
    for filename in os.listdir(directory):
        if not (filename.endswith('.dcm') or filename.endswith('.DCM')):
            continue
        dcm = dicom.read_file(os.path.join(directory, filename))
        summary_info = get_summary_info(dcm)
        lps_cm = image_position_patient_cm(dcm)
        # print(filename, "spacing", spacing_between_slices_cm(dcm), "pixel", pixel_spacing_cm(dcm), "lps_cm", lps_cm, "orientation", orientation)
        if between_slices_cm is None:
            between_slices_cm = spacing_between_slices_cm(dcm)
            pixel_cm = pixel_spacing_cm(dcm)
            pixel_x0_left_cm, pixel_y0_posterior_cm, _ = lps_cm
            orientation = image_orientation_patient(dcm)
        else:
            try:
                if between_slices_cm != spacing_between_slices_cm(dcm):
                    raise InvalidDataException("Inconsistent spacing between slices: %s and %s" % (between_slices_cm, spacing_between_slices_cm(dcm)))
                if pixel_cm != pixel_spacing_cm(dcm):
                    raise InvalidDataException("Inconsistent pixel spacing: %s and %s" % (pixel_cm, pixel_spacing_cm(dcm)))
                if [pixel_x0_left_cm, pixel_y0_posterior_cm] != lps_cm[:2]:
                    raise InvalidDataException("Inconsistent image position patient left posterior values: %s and %s" % (
                        [pixel_x0_left_cm, pixel_y0_posterior_cm], lps_cm[:2]
                    ))
                if orientation != image_orientation_patient(dcm):
                    raise InvalidDataException("Inconsistent image orientation patient values: %s and %s" % (
                        orientation, image_orientation_patient(dcm)
                    ))
                min_superior_cm = min(min_superior_cm, lps_cm[2]) if min_superior_cm is not None else lps_cm[2]
                max_superior_cm = max(max_superior_cm, lps_cm[2]) if max_superior_cm is not None else lps_cm[2]
            except InvalidDataException as e:
                summary_info['skip_reason'] = str(e)
                logging.warn(str(e))
                continue
        dicoms.append(dcm)
        dicom_summaries[filename] = summary_info
    logging.info("reading files -- took %s seconds" % (time.time() - before))

    before = time.time()
    bvalues = get_all_bvalues(dicoms)
    flip_angles = get_all_flipangles(dicoms)
    logging.info("checking bvalues and flip angles -- took {} seconds".format((time.time() - before)))

    all_series_descriptions = set([d.SeriesDescription for d in dicoms])
    # Still TODO: Ensure series description is the same for all files?
    # Same for mfr, study date, orientation, etc.
    results = {
        "manufacturer": dicoms[0].Manufacturer,
        "model": dicoms[0].ManufacturerModelName,
        # "series_description": dicoms[0].SeriesDescription,
        "study_date": get_datetime(dicoms[0]).isoformat(),
        "orientation": iop_string(image_orientation_patient(dicoms[0])),
        "description": "; ".join(all_series_descriptions),
        "image_coordinate_system": ImageCoordinateSystem(
            pixel_cm, between_slices_cm, orientation, pixel_x0_left_cm, pixel_y0_posterior_cm, min_superior_cm, max_superior_cm),
        "file_summary": dicom_summaries
    }

    # currently three valid cases:
    if len(bvalues) == 1 and len(flip_angles) > 1:
        # T1 case
        results["scalar_type"] = "T1"
        results["flip_angles_degrees"] = flip_angles
        results["alphas"] = np.array([np.deg2rad(angle) for angle in flip_angles])
        results["dwi"] = get_dwi_array(dicoms, "FLIP")
        results["rep_time_seconds"] = repetition_time(dicoms[0])
        return results

    elif len(bvalues) > 1 and len(flip_angles) == 1:
        # ADC case
        results["scalar_type"] = "ADC"
        results["bvalues"] = bvalues
        results["dwi"] = get_dwi_array(dicoms, "BVALUE")
        return results

    elif len(bvalues) == 1 and len(flip_angles) == 1:
        # thermometry case
        results["scalar_type"] = "thermometry"
        results["flip_angles_degrees"] = flip_angles
        results["dwi"] = get_dwi_array(dicoms, "FLIP")
        return results

    else:
        raise InvalidDataException("invalid data set with bvalues: {} and flip angles: {}".format(bvalues, flip_angles))
