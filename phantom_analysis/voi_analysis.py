# generalized VOI finding

import numpy as np
from copy import deepcopy
from collections import namedtuple, defaultdict
import cv2
import logging
# hungarian algorithm copied out of scipy
from hungarian import linear_sum_assignment


def get_vois(dwi_array, image_coordinate_system, phantom_definition, debug=False, name=None):
    """
    Find the VOIs for this phantom and ADC/T1.

    Args:
        dwi_array: 4d numpy array of dicom pixel data indexed by slice, then original dicom pixel_data x/y, then flip angle or b value;
             e.g. dwi[slice][x][y][flip_angle/b_value]
        image_coordinate_system: an ImageCoordinateSystem
        phantom_definition: phantom definition to use for VOI finding
        debug: optional parameter to show intermediate results

    Returns:
        VOIs as dictionary
    """
    vois = {}
    coronal_dwi = np.transpose(deepcopy(dwi_array), (3, 1, 0, 2))

    # Get the labeled ROIs
    found_circles = _find_all_circles_3D(coronal_dwi[0], phantom_definition, image_coordinate_system)
    roi_dict = _find_rois(found_circles, phantom_definition, image_coordinate_system)

    # construct coronal label map and then transpose at the end
    shape = np.shape(coronal_dwi[0])
    label_map = np.zeros(shape, dtype=int)

    # centers is dict: vial_label > list of centers (index is coronal slice), center is None if slice doesn't have the VOI
    centers = {}
    # found_vois is dict: vial_label > dict with center, height, volume, and radius
    found_vois = {}
    for i in range(1, phantom_definition["config"]["expected_total_number_vois"] + 1, 1):
        centers[i] = [None] * shape[0]
        found_vois[i] = {}
    # error per coronal slice, 0 if slice didn't contain expected ROIs
    error = [0] * shape[0]

    coronal_slice_to_circles = roi_dict["rois"]
    for s in coronal_slice_to_circles:
        # for slices that didn't have expected ROIs
        if coronal_slice_to_circles[s] == {}:
            # label map is already set to zeros
            # don't need to look for circles or increment error
            continue
        # otherwise create label map for this coronal slice
        slice_labels = np.zeros(np.shape(label_map[0]))
        slice_err = coronal_slice_to_circles[s]["error"]
        # Currently thresholding on our error values
        if slice_err < phantom_definition["config"]["max_error"]:
            for v in coronal_slice_to_circles[s]["circles"]:
                center_x, center_y, radius = coronal_slice_to_circles[s]["circles"][v]
                # save the center and radius (in cm)
                centers[v][s] = (center_x, center_y, radius)
                # add this circle to slice label map (in pixels)
                cv2.ellipse(slice_labels,
                            center=image_coordinate_system.coronal_center_to_pixels(center_x, center_y),
                            axes=(int(np.rint(radius/image_coordinate_system.pixel_spacing_cm)), int(np.rint(radius/image_coordinate_system.spacing_between_slices_cm)) ),
                            angle=0.0,
                            startAngle=0.0,
                            endAngle=360.0,
                            color=v, thickness=-1)

            #_show_intermediate(coronal_dwi[0][s], coronal_slice_to_circles[s]["circles"].values(), image_coordinate_system)
        error[s] = slice_err
        label_map[s] = slice_labels

        # save images with circles if debug is True
        if debug:
            _show_intermediate(coronal_dwi[0][s], coronal_slice_to_circles[s]["circles"].values(), image_coordinate_system, name=name, slice=s)

    # get the center, height, and radius of each VOI
    for v in centers:
        all_centers = [ x for x in centers[v] if x is not None ]
        if not all_centers:
            continue

        # Get first and last slices of VOI
        first_slice, last_slice = None, None
        for s in range(len(centers[v])):
            if centers[v][s] is not None:
                if first_slice is None:
                    first_slice = s
                    last_slice = s
                elif last_slice is not None:
                    last_slice = s

        #print("voi {}".format(v))
        #print("first slice: {}, cm {}".format(first_slice, image_coordinate_system.coronal_z_to_cm(first_slice)))
        #print("last slice: {}, cm {}".format(last_slice, image_coordinate_system.coronal_z_to_cm(last_slice)))
        height_cm = image_coordinate_system.pixel_spacing_cm * (last_slice - first_slice)
        coronal_z_center_cm = image_coordinate_system.coronal_z_to_cm( first_slice + ((last_slice - first_slice)/2) )
        coronal_x_center_cm = np.mean([x[0] for x in all_centers])
        coronal_y_center_cm = np.mean([x[1] for x in all_centers])
        avg_radius = np.mean([x[2] for x in all_centers])
        # flip to original coordinates at this point
        found_vois[v]["center"] = [ coronal_x_center_cm, coronal_z_center_cm, coronal_y_center_cm ]
        found_vois[v]["radius"] = avg_radius
        found_vois[v]["height"] = height_cm
        found_vois[v]["coronal_center_std_dev"] = [ np.std([x[0] for x in all_centers]), np.std([x[1] for x in all_centers]) ]

    label_map = np.transpose(label_map, (1,0,2))
    vois["label_map"] = label_map
    vois["roi_centers"] = centers
    vois["error_by_coronal_slice"] = error
    vois["found_vois"] = found_vois
    vois["total_error"] = roi_dict["error"]
    vois["transform"] = roi_dict["transform"]
    return vois

def _find_rois(found_circles_3D, phantom_definition, image_coordinate_system):
    flip_180 = np.array([[-1, 0, 0, 0],[0, -1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
    transforms = [None, flip_180] # this is where we'll put logic to find best transform

    min_error = None
    best_rois = {}
    best_t = None
    for t in transforms:
        rois, error = _3D_graph_match(found_circles_3D, phantom_definition, image_coordinate_system, t)
        #print("transform: {}, error: {}".format(t, error))
        if min_error is None:
            min_error = error
            best_rois = rois
            best_t = t
        elif error < min_error:
            min_error = error
            best_rois = rois
            best_t = t

    logging.info("best transform: {}".format(best_t))
    return { "transform": best_t, "error": min_error, "rois": best_rois }

def _3D_graph_match(found_circles_3D, phantom_definition, image_coordinate_system, transform=None):
    volume_error = 0
    coronal_slice_to_rois = {}

    # get the coronal_heights (in cm) for each coronal slice
    coronal_heights_cm = [ image_coordinate_system.coronal_z_to_cm(coronal_slice) for coronal_slice in range(len(found_circles_3D))]
    for s, z_cm, found_rois in zip(range(len(coronal_heights_cm)), coronal_heights_cm, found_circles_3D):
        expected_rois = _get_expected_circles_by_coronal_slice(z_cm, phantom_definition, transform)
        # skip any slice where there aren't expected rois
        if expected_rois == {}:
            #print("skip slice {}, height {}".format(s, z_cm))
            continue

        #print("number of found circles on slice {}: {}".format(s, len(found_rois)))
        matched_rois, slice_err = _graph_match(found_rois, expected_rois, phantom_definition)
        volume_error += slice_err
        slice_vals = { "circles": matched_rois, "error": slice_err}
        coronal_slice_to_rois[s] = slice_vals

    return coronal_slice_to_rois, volume_error

def _find_all_circles_3D(coronal_dwi_array, phantom_definition, image_coordinate_system):
    # build up a 2D array of potential ROIs per coronal slice (in cm)
    found_circles_3D = [None] * np.shape(coronal_dwi_array)[0]

    for coronal_slice, dwi in enumerate(coronal_dwi_array):
        if np.max(dwi) - np.min(dwi) < 8:
            # image doesn't contain enough information to warrant looking for VOIs
            # also _map_uint16_to_uint8 doesn't support min equaling the max
            continue

        cm_circles = _find_circles_in_coronal_slice(dwi, phantom_definition["config"], image_coordinate_system, coronal_slice)
        found_circles_3D[coronal_slice] = cm_circles
        #_show_intermediate(dwi, cm_circles, image_coordinate_system, name="cm_circles", slice=coronal_slice)

    return found_circles_3D

# find and return all circles in a pixel array (single coronal slice)
def _find_circles_in_coronal_slice(pixel_array, config_params, image_coordinate_system, slice_num):
    cm_per_pixel = image_coordinate_system.pixel_spacing_cm
    cm_per_slice = image_coordinate_system.spacing_between_slices_cm
    scale = cm_per_pixel/cm_per_slice
    dim = pixel_array.shape
    new_shape = ( int(round(dim[0]/scale)), dim[1] ) if scale < 1 else ( dim[0], int(round(dim[1]*scale)) )
    cm_per_pixel = min(cm_per_pixel, cm_per_slice)

    if new_shape != dim:
        # numpy shape and opencv x,y are reversed
        pixel_array = cv2.resize(pixel_array, (new_shape[1], new_shape[0]))


    #print("Before blurring"); _show_intermediate(pixel_array)
    blurred_img = _map_uint16_to_uint8(pixel_array)
    #print("After blurring"); _show_intermediate(blurred_img)
    #print("blurred img shape: {}".format(np.shape(blurred_img)))

    # CLAHE (Contrast Limited Adaptive Histogram Equalization) then Gaussian Blur
    # CLAHE is not documented, see http://stackoverflow.com/a/24341809/1007353 for some explanation
    clahe = cv2.createCLAHE(clipLimit=config_params["clahe_limit"], tileGridSize=config_params["clahe_grid_size"])
    blurred_img = clahe.apply(blurred_img)

    # gaussian blur seems both better at finding circles and finding more precise centers than both average and median
    # blurs.  gaussian also better at finding true circles than bilateralFilter
    blurred_img = cv2.GaussianBlur(blurred_img, config_params["gaussian_blur_kernel_size"], 0)
    #("After CLAHE and gaussian blur"); _show_intermediate(blurred_img)

    cv2_circles = cv2.HoughCircles(blurred_img,
                                   cv2.HOUGH_GRADIENT,
                                   config_params["dp"],
                                   int(round(config_params["min_distance_between_circle_centers"]/cm_per_pixel)),
                                   param1=config_params["canny_threshold"],
                                   param2=config_params["acc_threshold"],
                                   minRadius=int(round(config_params["min_radius"]/cm_per_pixel)),
                                   maxRadius=int(round(config_params["max_radius"]/cm_per_pixel)))

    if cv2_circles is None:
        return []

    #_show_intermediate(blurred_img, cv2_circles[0], name="all_circles", slice=slice_num)
    cv2_circles = np.array([[image_coordinate_system.pixel_x0_left_cm - c[0],
                             image_coordinate_system.max_superior_cm - c[1],
                             c[2]] for c in cv2_circles[0]*cm_per_pixel])
    return cv2_circles

# Get the expected ROI labels, centers, and radii for a coronal height and transform
def _get_expected_circles_by_coronal_slice(coronal_height_cm, phantom_definition, transform=None):
    expected_rois = {}
    for label,params in phantom_definition["vois"].items():
        # move center based on transform, then use new center to calculate expected ROIs
        center = params["center_cm"]
        if transform is not None:
            # transforms in 3D are a 4x4 matrix (this allows for combination of translation and rotation) -- we should only have translation (3D) and 2D rotation
            vector = np.reshape(np.array([center[0], center[1], center[2], 1.0]), (4,1))
            new_center = np.matmul(transform, vector)
            center = np.reshape(new_center, (1,4))[0][0:3]

        # TODO: If a transform moved the expected rois out of the volume, that should be handled
        #          by the 3D graph match being based on slices of the found circles volume. It would result in
        #          there not being expected rois in the actual volume, rather than searching for found circles
        #          outside of the volume. Problem is this would lead to a minimization of the error.
        #          If we wanted to do a check that we're in the valid region after the transform we would need the
        #          coordinates in cm for the entire volume (maybe add this as a function in image_coordinate_system).
        #          Handle this when creating viable transforms.

        z = center[2]
        # Get distance from center that this VOI extends (assumes sphere if not cylinder)
        half_height = params["height_cm"]/2 if params["shape"] == "cylinder" else params["radius_cm"]
        # Limit height by maximum height percentage
        half_height = half_height * phantom_definition["config"]["height_max_percentage"]

        if coronal_height_cm < z - half_height or coronal_height_cm > z + half_height:
            # not on a slice in this VOI (check this after the transform to work with z transforms)
            continue
        radius = params["radius_cm"]
        if params["shape"] == "sphere":
            expected_radius = np.sqrt(np.abs(np.power(radius, 2) - np.power((coronal_height_cm - z), 2)))
            roi = [center[0], center[1], expected_radius]
        elif params["shape"] == "cylinder":
            roi = [center[0], center[1], radius]

        expected_rois[label] = roi

    return expected_rois    # in cm

def _graph_match(found_circles_cm, expected_rois, phantom_definition):
    # if found_circles_cm is None or [] accumulate error (doesn't get called if expected_rois is {})
    if found_circles_cm is None or found_circles_cm == []:
        slice_error = len(expected_rois) * phantom_definition["config"]["graph_match_threshold_factor"] * phantom_definition["vois"][1]["radius_cm"]
        return {}, slice_error

    # create the cost matrix
    distance_matrix = np.zeros((len(expected_rois), len(found_circles_cm)))
    expected_roi_labels = expected_rois.keys()
    for i, expected_circle_label in enumerate(expected_roi_labels):
        expected_circle = expected_rois[expected_circle_label]
        for found_circle_index, found_circle in enumerate(found_circles_cm):
            distance = cv2.norm(np.array(expected_circle[0:2], dtype=found_circle.dtype), found_circle[0:2])
            distance_matrix[i][found_circle_index] = distance

    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    results = zip(row_ind, col_ind)

    # unpack these into the format we want
    matched_rois = {}
    slice_error = 0
    for expected_index, found_index in results:
        distance = distance_matrix[expected_index, found_index]
        # threshold on distance between expected and found circles
        if distance < phantom_definition["config"]["graph_match_threshold_factor"] * phantom_definition["vois"][1]["radius_cm"]:
            found_circle = found_circles_cm[found_index]
            # bound the radius
            max_radius_cm = expected_rois[expected_roi_labels[expected_index]][2] * phantom_definition["config"]["radius_max_percentage"]
            found_circle[2] = min(found_circle[2], max_radius_cm)
            matched_rois[expected_roi_labels[expected_index]] = found_circle
            slice_error += distance

    slice_error += (len(expected_rois) - len(matched_rois)) * phantom_definition["config"]["graph_match_threshold_factor"] * phantom_definition["vois"][1]["radius_cm"]

    return matched_rois, slice_error


_graph_match._method_logged = False


def _map_uint16_to_uint8(img, lower_bound=None, upper_bound=None):
    """
    Map a 16-bit image through a lookup table to convert it to 8-bit.

    This is used because OpenCV transforms like HoughCircles and CLAHE require 8-bit.

    Args:
        img: numpy.ndarray[np.uint16]
            image that should be mapped
        lower_bound: int, optional
            lower bound of the range that should be mapped to ``[0, 255]``,
            value must be in the range ``[0, 65535]`` and smaller than `upper_bound`
            (defaults to ``numpy.min(img)``
)        upper_bound: int, optional
           upper bound of the range that should be mapped to ``[0, 255]``,
           value must be in the range ``[0, 65535]`` and larger than `lower_bound`
           (defaults to ``numpy.max(img)``)

    Returns: numpy.ndarray[uint8]
    """
    if not(0 <= lower_bound < 2**16) and lower_bound is not None:
        raise ValueError(
            '"lower_bound" must be in the range [0, 65535]')
    if not(0 <= upper_bound < 2**16) and upper_bound is not None:
        raise ValueError(
            '"upper_bound" must be in the range [0, 65535]')
    if lower_bound is None:
        lower_bound = np.min(img)
    if upper_bound is None:
        upper_bound = np.max(img)
    if lower_bound >= upper_bound:
        raise ValueError('"lower_bound" must be smaller than "upper_bound" (lower_bound=%s, upper_bound=%s)' % (
            lower_bound, upper_bound))
    lut = np.concatenate([
        np.zeros(lower_bound, dtype=np.uint16),
        np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
        np.ones(2**16 - upper_bound, dtype=np.uint16) * 255
    ])
    return lut[img].astype(np.uint8)


def _show_intermediate(img, cv2_circles=None, image_coordinate_system=None, expected_centers=None, name=None, slice=None):
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if cv2_circles is not None:
        if image_coordinate_system is not None:
            for i, (x, y, r) in enumerate(cv2_circles, start=1):
                print("circle with index %s: %s, %s, %s" % (i, x, y, r))
                center = image_coordinate_system.coronal_center_to_pixels(x, y)
                print("circle with center {} and radius {}, {}".format(center, int(np.rint(r/image_coordinate_system.pixel_spacing_cm)), int(np.rint(r/image_coordinate_system.spacing_between_slices_cm))))
                cv2.ellipse(img, center, ((int(np.rint(r/image_coordinate_system.pixel_spacing_cm)), int(np.rint(r/image_coordinate_system.spacing_between_slices_cm)) )), 0, 0, 360, (0, 255, 0), 1)
        else:
            for i, (x, y, r) in enumerate(cv2_circles, start=1):
                print("circle with index %s: %s, %s, %s" % (i, x, y, r))
                cv2.circle(img, (x, y), r, (0, 255, 0), 1)
                #cv2.putText(img, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
    if expected_centers is not None:
        for (x,y,r) in expected_centers:
            center = image_coordinate_system.coronal_center_to_pixels(x, y)
            cv2.circle(img, center, 1, (0, 255, 0), -1)
            print("expected center: {},{} -> {}".format(x, y, center))

    if name is not None and slice is not None:
        cv2.imwrite("%s-%04d.png" % (name, slice), img);
    else:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(13, 13))
        plt.imshow(img)
        plt.show()
