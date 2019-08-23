#!/usr/bin/env python

import os
import argparse
import yaml
import sys
from collections import defaultdict
import logging
import numpy as np
import platform

from phantom_analysis import dicom_util, scalar_analysis, voi_analysis, phantom_definitions

WINDOWS = True if platform.system() == 'Windows' else False
CLAMP = (0, 4000)

def isclose(a, b, rel_tol=1e-06, abs_tol=1e-3):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def fuzzy_compare_dict(a, b):
    if len(a) != len(b):
        return False
    for k in a.keys():
        if k not in b:
            return False

        if isinstance(a[k], dict):
            if not (isinstance(b[k], dict) and fuzzy_compare_dict(a[k], b[k])):
                return False
        elif isinstance(a[k], float):
            if not (isinstance(b[k], float) and isclose(a[k], b[k])):
                return False
        else:
            if not a[k] == b[k]:
                return False
    return True

def rounded_float_representer(dumper, value):
    text = '{0:.4f}'.format(value)
    return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)

def baseline_format(expected, dir_output, voi_output, phantom_definition):
    baseline = defaultdict(lambda: defaultdict)  # 2 keys deep
    center_distances = []
    volume_differences = []
    scalar_median_differences = []
    scalar_mean_differences = []
    total_clamped_pixels = 0

    # Calculate scalar maps and VOI stats
    scalar_type = dir_output["scalar_type"]
    label_map = voi_output["label_map"]
    if scalar_type == "ADC":
        scalar_map = scalar_analysis.calculate_adc(dir_output["dwi"], dir_output["bvalues"]) * 1e6
        voi_stats = scalar_analysis.voi_stats(label_map, scalar_map, dir_output["image_coordinate_system"], phantom_definition)
    elif scalar_type == "T1":
        scalar_map = scalar_analysis.calculate_t1(dir_output["dwi"], dir_output["alphas"], dir_output["rep_time_seconds"], use_pool= not WINDOWS, clamp=CLAMP, threshold=5)
        voi_stats = scalar_analysis.voi_stats(label_map, scalar_map, dir_output["image_coordinate_system"], phantom_definition, clamp=CLAMP)
    else:
        # Shouldn't ever get here
        return baseline

    # Analyze stats for each VOI
    vois = voi_output["found_vois"]
    for voi in vois:
        expected_voi_dict = expected[voi]
        baseline[voi] = {}

        # Compare center coordinates and variation in center
        if "center" in vois[voi]:
            center_info = {
                "center_left_cm": vois[voi]["center"][0],
                "center_posterior_cm": vois[voi]["center"][1],
                "center_superior_cm": vois[voi]["center"][2],
                "std_dev_of_coronal_center": { "coronal x": vois[voi]["coronal_center_std_dev"][0], "coronal y": vois[voi]["coronal_center_std_dev"][1] }
                }
            baseline[voi].update(center_info)

            # Compare found center to expected (if known)
            if "center_left_cm" in expected_voi_dict:
                center_distance_cm = np.linalg.norm(np.array([expected_voi_dict["center_left_cm"],
                                                              expected_voi_dict["center_posterior_cm"],
                                                              expected_voi_dict["center_superior_cm"]]) - vois[voi]["center"])
                center_distances.append(center_distance_cm)
                baseline[voi].update({ "center_distance_cm": center_distance_cm })

        # Compare scalar stats to expected values for this dataset (ADC/T1)
        scalar_type = scalar_type.lower()
        if voi in voi_stats:
            median_diff = expected_voi_dict["{}_median".format(scalar_type)] - voi_stats[voi]["median"]
            mean_diff = expected_voi_dict["{}_mean".format(scalar_type)] - voi_stats[voi]["mean"]
            scalar_median_differences.append(np.abs(median_diff))
            scalar_mean_differences.append(np.abs(mean_diff))
            total_clamped_pixels += voi_stats[voi]["clamped_pixels"]
            volume_difference_percent = (voi_stats[voi]["volume"] - expected_voi_dict["volume_cm3"])/expected_voi_dict["volume_cm3"] * 100
            volume_differences.append(volume_difference_percent)

            baseline[voi].update({
                "{}_median".format(scalar_type): voi_stats[voi]["median"],
                "{}_median_difference".format(scalar_type): median_diff,
                "{}_mean".format(scalar_type): voi_stats[voi]["mean"],
                "{}_mean_difference".format(scalar_type): mean_diff,
                "{}_max".format(scalar_type): voi_stats[voi]["max"],
                "{}_min".format(scalar_type): voi_stats[voi]["min"],
                "{}_std_dev".format(scalar_type): voi_stats[voi]["std_dev"],
                "number_clamped_pixels": voi_stats[voi]["clamped_pixels"],
                "volume_cm3": voi_stats[voi]["volume"],
                "volume_difference_percent": volume_difference_percent
                })

    # Gather information about entire dataset
    missing_voi_count = len(expected.keys()) - len(voi_stats.keys())
    baseline.update({
        "center_max_distance_cm": max(center_distances) if center_distances else 0.0,
        "center_average_distance_cm": np.mean(center_distances) if center_distances else 0.0,
        "volume_max_difference_percent": max(volume_differences) if volume_differences else 0.0,
        "volume_average_difference_percent": np.mean(volume_differences) if volume_differences else 0.0,
        "missing_voi_count": missing_voi_count,
        "{}_median_max_difference".format(scalar_type): max(scalar_median_differences) if scalar_median_differences else 0.0,
        "{}_median_average_difference".format(scalar_type): np.mean(scalar_median_differences) if scalar_median_differences else 0.0,
        "{}_mean_max_difference".format(scalar_type): max(scalar_mean_differences) if scalar_mean_differences else 0.0,
        "{}_mean_average_difference".format(scalar_type): np.mean(scalar_mean_differences) if scalar_mean_differences else 0.0,
        "total_clamped_pixels": total_clamped_pixels,
        "error_in_voi_finding_for_dataset": voi_output["total_error"]
    })

    return baseline

def test_voi(update_baseline=False):
    """
    Compare volume of interest finding against previously saved baselines.

    Args:
        update_baseline: update baseline files if actual results don't match

    Returns:
        Number of test failures (returns zero if everything passes).
    """
    # simplify yaml representation to be more human readable and to round floats for simpler comparison
    yaml.add_representer(defaultdict, yaml.representer.Representer.represent_dict)
    yaml.add_representer(np.float64, rounded_float_representer)
    yaml.add_representer(np.float32, rounded_float_representer)
    yaml.add_representer(float, rounded_float_representer)

    # Get test cases
    test_voi_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data", "voi")
    all_cases = os.listdir(os.path.join(test_voi_dir, "adc")) + os.listdir(os.path.join(test_voi_dir, "t1"))

    # Initialize counters and arrays
    test_cases = 0
    fail_count = 0
    missing_voi_per_test_case = []
    average_center_distance_per_test_case = []
    max_center_distance = 0
    average_volume_difference_per_test_case = []
    max_volume_difference = 0
    average_t1_median_difference_per_test_case = []
    max_t1_median_difference = 0
    average_t1_mean_difference_per_test_case = []
    max_t1_mean_difference = 0
    average_adc_median_difference_per_test_case = []
    max_adc_median_difference = 0
    average_adc_mean_difference_per_test_case = []
    max_adc_mean_difference = 0

    # Process each test case
    for test_case in sorted(all_cases):
        logging.info("Test case %s" % test_case)
        test_cases += 1

        # Parse name of directory
        # Naming scheme for test directories is model_num_t1shape_scalar_date_scan
        dir_name_parts = test_case.upper().split("_")
        if len(dir_name_parts) != 6:
            logging.warn("test case {} incorrectly formatted".format(test_case))
            continue            
        model_num = dir_name_parts[0] + "_" + dir_name_parts[1]
        t1_vial_shape = dir_name_parts[2]
        scalar_type = dir_name_parts[3]

        # Read directory
        test_case_dir = os.path.join(test_voi_dir, scalar_type.lower(), test_case)
        try:
            dir_output = dicom_util.read_dicomdir(test_case_dir)
        except Exception:
            logging.warn("Failed to read DICOM files.")
            raise

        # Get flag for debug in get_vois()
        debug = float(os.environ.get("PHANTOM_ANALYSIS_SHOW_INTERMEDIATE_RESULTS", "0")) != 0.0
        # Check scalar type and get phantom definition for this dataset
        assert dir_output["scalar_type"] == scalar_type
        phantom_def_name = model_num + "_" + t1_vial_shape + "_" + scalar_type
        phantom_def = getattr(phantom_definitions, phantom_def_name)
        # Get vois for this dataset and phantom definition
        voi_output = voi_analysis.get_vois(dir_output["dwi"], dir_output["image_coordinate_system"], phantom_def, debug, name=test_case)

        # Get baseline and expected files for this dataset
        baseline_path = os.path.join(test_case_dir, "baseline.yaml")
        expected_path = os.path.join(test_case_dir, "target.yaml")

        with open(expected_path, 'r') as expected_file:
            expected = yaml.safe_load(expected_file)

        if os.path.exists(baseline_path):
            with open(baseline_path, 'r') as baseline_file:
                baseline = yaml.safe_load(baseline_file)
        else:
            baseline = None

        # compare baseline to vois after formatting through yaml to ensure same floating point precision
        actual = yaml.safe_load(yaml.dump(baseline_format(expected, dir_output, voi_output, phantom_def)))
        if not fuzzy_compare_dict(actual, baseline):
            logging.warn("\tFAIL")
            fail_count += 1
            if update_baseline:
                logging.info("\tupdating baseline")
                with open(baseline_path, 'w') as baseline_file:
                    yaml.dump(actual, baseline_file)
            else:
                actual_path = "actual_%s.yaml" % test_case
                with open(actual_path, 'w') as actual_file:
                    yaml.dump(actual, actual_file)
                logging.warn("\tactual results don't match baseline: compare %s %s" % (baseline_path, actual_path))
        else:
            logging.info("\tPASS")

        # track summary info
        missing_voi_per_test_case.append(actual["missing_voi_count"])
        average_center_distance_per_test_case.append(actual["center_average_distance_cm"])
        max_center_distance = max(max_center_distance, actual["center_max_distance_cm"])
        average_volume_difference_per_test_case.append(actual["volume_average_difference_percent"])
        max_volume_difference = max(max_volume_difference, actual["volume_max_difference_percent"])
        if dir_output["scalar_type"] == "T1":
            average_t1_median_difference_per_test_case.append(actual["t1_median_average_difference"])
            max_t1_median_difference = max(max_t1_median_difference, actual["t1_median_max_difference"])
            average_t1_mean_difference_per_test_case.append(actual["t1_mean_average_difference"])
            max_t1_mean_difference = max(max_t1_mean_difference, actual["t1_mean_max_difference"])
        elif dir_output["scalar_type"] == "ADC":
            average_adc_median_difference_per_test_case.append(actual["adc_median_average_difference"])
            max_adc_median_difference = max(max_adc_median_difference, actual["adc_median_max_difference"])
            average_adc_mean_difference_per_test_case.append(actual["adc_mean_average_difference"])
            max_adc_mean_difference = max(max_adc_mean_difference, actual["adc_mean_max_difference"])

    summary = {
        "test_cases": test_cases,
        "voi_missing_max": max(missing_voi_per_test_case),
        "voi_missing_average": np.mean(missing_voi_per_test_case),
        "voi_center_max_distance_cm": max_center_distance,
        "voi_center_average_distance_cm": np.mean(average_center_distance_per_test_case),
        "voi_volume_max_difference_percent": max_volume_difference,
        "voi_volume_average_difference_percent": np.mean(average_volume_difference_per_test_case),
        "t1_median_max_difference": max_t1_median_difference,
        "t1_median_average_difference": np.mean(average_t1_median_difference_per_test_case),
        "t1_mean_max_difference": max_t1_mean_difference,
        "t1_mean_average_difference": np.mean(average_t1_mean_difference_per_test_case),
        "adc_median_max_difference": max_adc_median_difference,
        "adc_median_average_difference": np.mean(average_adc_median_difference_per_test_case),
        "adc_mean_max_difference": max_adc_mean_difference,
        "adc_mean_average_difference": np.mean(average_adc_mean_difference_per_test_case)
    }
    with open(os.path.join(test_voi_dir, "summary.yaml"), 'w') as summary_file:
        yaml.dump(summary, summary_file)

    return fail_count


def main():
    parser = argparse.ArgumentParser(description="Test if voi calculations match saved baseline")
    parser.add_argument('-u', '--update-baseline', action='store_true',
                        help="Write new baseline if baseline missing or doesn't match")
    args = parser.parse_args()

    failures = test_voi(args.update_baseline)

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logging.getLogger().handlers[0].setFormatter(formatter)
    main()
