import numpy as np
import time
import multiprocessing as mp
from functools import partial
import logging

if tuple(np.version.version.split(".")[:2]) >= ('1', '14'):
    _lstsq = partial(np.linalg.lstsq, rcond=None)
else:
    _lstsq = np.linalg.lstsq

POLYFIT_SIZE=1024*256

def calculate_adc(dwi_array, bvals_array):
    """
    Calculate the ADC at each point and return the ADC volume.

    Args:
        dwi_array: 4d numpy array of dicom pixel data indexed by slice, then dicom x,y, then b value
            e.g. dwi_array[slice][x][y][b_value]
        bvals_array: array containing the b values for the volume

    Returns:
        3d numpy array of ADC values for volume indexed by slice, then dicom x,y

    """
    z,y,x,b = np.shape(dwi_array)


    THRESHOLD_FRACTION = 0.004
    threshold = int(THRESHOLD_FRACTION * np.max(dwi_array))

    y_vals = np.transpose(dwi_array)
    y_vals = np.reshape(y_vals, (b, x*y*z))

    # threshold the values
    y_vals[y_vals <= threshold] = 0

    # create masked array for log values to handle log(0) cases
    logy = np.ma.log(y_vals)

    # Free y_vals
    y_vals = None

    mask = np.ma.getmask(logy)
    logy.filled(0)

    # calculate ADC for each point in volume
    # do the polyfit in chunks to save memory
    adc_map = np.zeros(logy.shape[1])
    for i in range(1+logy.shape[1]/(POLYFIT_SIZE)):
        s = i*POLYFIT_SIZE
        e = min((i+1)*POLYFIT_SIZE, logy.shape[1])
        if s == e:
            break
        # ADC is first coeff in polyfit solve
        adc_map[s:e] = np.polyfit(bvals_array, logy[:, s:e], 1)[0]

    # get rid of values where log(y) would've been nan
    for i in range(np.shape(mask)[0]):
        adc_map[mask[i] == True] = 0

    adc_map = np.reshape(adc_map, (x, y, z))
    adc_map = np.transpose(adc_map)

    # clean up ADC values
    adc_map *= -1
    adc_map[np.isnan(adc_map)] = 0
    adc_map[adc_map < 0] = 0

    return adc_map

# TODO: consider updating unit testing to include clamp, threshold cases
def calculate_t1_point(values, alphas, tr, clamp=None, threshold=None):
    """
    Calculates the T1 value for a single point with log weighting using the alphas and the matching data values.

    Approach described in Linear least square T1 calculation described by section 22.4.2
        "T1 Estimation from SSI Measurements at Multiple Flip Angles" in "Magnetic Resonance Imaging: Physical Principles and Sequence Design".

    Args:
        values:the values from the dicom files of the point (one value per flip angle)
        alphas: the flip angles in radians from dicom files
        tr: repetition time from dicom files
        clamp: min and max for valid T1 values
        threshold: minimum difference in values for valid calculation

    Returns:
        the T1 value for this point
    """
    if threshold is not None:
        if np.sum(values) < len(values)*threshold:
            return 0
    a = np.argsort(alphas)
    x = values[a]/np.sin(alphas[a])
    y = values[a]/np.tan(alphas[a])

    weight = 1.1
    A = np.vstack([x, np.ones(len(x))]).T
    W = np.sqrt(np.diag(np.logspace(weight,0,y.shape[0])))
    A = np.dot(W,A)
    y = np.dot(y,W)

    slope, intercept = _lstsq(A,y)[0]
    # TODO: is it appropriate to set negative values to 0? Why are we getting negative values?
    if slope <= 0:
        return 0

    t1 = tr / np.log(slope)
    if clamp:
        t1 = max(clamp[0], min(clamp[1], t1))
    return t1

def calculate_t1(dwi_array, alphas, tr, use_pool=True, clamp=None, threshold=None):
    """
    Calculate the T1 map for a volume.

    Args:
        dwi_array: 4d numpy array of dicom pixel data indexed by slice, then dicom x,y, then flip angle
            e.g. dwi_array[slice][x][y][flip_angle]
        alphas: the flip angles in radians from dicom files
        tr: repetition time from dicom files
        use_pool: use pool for multiprocessing to speed up calculation
        clamp = min and max values for T1
        threshold: minimum difference in values for valid calculation

    Returns:
        3d numpy array of the T1 values for the volume indexed by slice, then x,y
    """
    logging.info("starting T1 calculation")
    before = time.time()
    z, y, x, f = np.shape(dwi_array)

    values_array = np.reshape(dwi_array, (z*y*x, f))
    unique_values, inverse = np.unique(values_array, axis=0, return_inverse=True)
    logging.info("Found {} unique t1 values in {:0.3f}s".format(unique_values.shape[0], time.time()-before)); before=time.time()

    # create the pool of processes
    t1_fixed_alpha = partial(calculate_t1_point, alphas=alphas, tr=tr, clamp=clamp, threshold=threshold)

    if use_pool:
        logging.info("Using pool")
        pool = mp.Pool()
        t1_values = np.array(pool.map(t1_fixed_alpha, unique_values))
        pool.close()
        pool.join()
    else:
        t1_values = np.array(map(t1_fixed_alpha, unique_values))

    t1 = np.reshape(t1_values[inverse], (z,y,x))
    logging.info("Completed t1 calculation in {:0.3f}s".format(time.time()-before)); before=time.time()
    return t1

# Alternative way to calculate T1 map for an entire volume
# This is slightly faster on large volumes, but slower on single slices
def t1_slice(dwi_slice, alphas, tr):
    y,x,f = np.shape(dwi_slice)
    values_array = np.reshape(dwi_slice, (y*x, f))
    t1_slice = [ calculate_t1_point(v, alphas, tr) for v in values_array ]
    t1_slice = np.reshape(t1_slice, (y, x))
    return t1_slice

def t1_map_by_slice(dwi_array, alphas, tr):
    z,y,x,f = np.shape(dwi_array)

    # create the pool of processes
    pool = mp.Pool()
    slice_fixed_alpha = partial(t1_slice, alphas=alphas, tr=tr)
    t1 = pool.map(slice_fixed_alpha, dwi_array)

    pool.close()
    pool.join()

    return t1

# TODO: unit testing doesn't include optional clamp parameter
def voi_stats(label_map, scalar_map, image_coordinate_system, phantom_definition, clamp=None):
    """
    Calculate the VOI stats for a scalar map.

    Args:
        label_map: 3d numpy array with integer labels for the VOIs
        scalar_map: 3d numpy array with scalar values (ADC/T1)
        image_coordinate_system: image coordinate system for the dataset
        phantom_definition: phantom definition for the dataset
        clamp = min and max values for T1

    Returns:
        Dictionary from VOI label to stats
    """
    voi_stats = {}
    pixel_volume = image_coordinate_system.pixel_spacing_cm * image_coordinate_system.pixel_spacing_cm * image_coordinate_system.spacing_between_slices_cm

    voi_labels = np.unique(label_map)[1:]
    for i in voi_labels:
        # get the points for this VOI
        voi_points_z, voi_points_x, voi_points_y = np.nonzero(label_map == i)
        # get scalar values for this VOI
        scalar_values = scalar_map[voi_points_z, voi_points_x, voi_points_y]
        voi_num_pixels = len(scalar_values)
        if clamp is not None:
            scalar_values = scalar_values[scalar_values > clamp[0]]
            scalar_values = scalar_values[scalar_values < clamp[1]]
        # return stats on this VOI
        voi_def = phantom_definition["vois"][i]
        mean = np.mean(scalar_values)
        median = np.mean(scalar_values)
        expected_value = voi_def["expected_value"]

        voi_stats[i] = { "median": median,
                         "median_percent_diff": (median - expected_value)/expected_value * 100 if expected_value is not None else None,
                         "mean": np.mean(scalar_values),
                         "mean_percent_diff": (mean - expected_value)/expected_value * 100 if expected_value is not None else None, 
                         "max": np.amax(scalar_values),
                         "std_dev": np.std(scalar_values),
                         "clamped_pixels": voi_num_pixels - len(scalar_values),
                         "label": i,
                         "content_type": voi_def["content_type"],
                         "expected_value": expected_value, 
                         "min": np.amin(scalar_values),
                         "count": voi_num_pixels,
                         "volume": voi_num_pixels * pixel_volume
                        }
    return voi_stats
