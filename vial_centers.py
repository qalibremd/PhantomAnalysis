#! /usr/bin/env python

import numpy as np
# get the centers for the vials

adc_inner_vials = 4
adc_outer_vials = 12
adc_inner_offset = 45
adc_outer_offset = 15
adc_inner_radius = 2.2
adc_outer_radius = 4.4

adc_center = [-9.145, 0.5588]
# S = 5.588 mm, R = 91.451mm
t1_center = [9.6, 0.34]

t1_small = 3
t1_large = 8
t1_small_offset = 30
t1_small_radius = 1.3
t1_large_radius = 3.0


if __name__ == "__main__":
	print("ADC inner vials")
	angle = 360 / adc_inner_vials
	for i in range(adc_inner_vials):
		degree_angle = adc_inner_offset + i*angle
		print("vial {}: x: {}, y: {}".format(i + 1, \
			adc_center[0] - np.cos(np.radians(degree_angle)) * adc_inner_radius, \
			adc_center[1] + np.sin(np.radians(degree_angle)) * adc_inner_radius))

	print("ADC outer vials")
	angle = 360 / adc_outer_vials
	for i in range(adc_outer_vials):
		degree_angle = adc_outer_offset + i*angle
		print("vial {}: x: {}, y: {}".format(i + 5, \
			adc_center[0] - np.cos(np.radians(degree_angle)) * adc_outer_radius, \
			adc_center[1] + np.sin(np.radians(degree_angle)) * adc_outer_radius))


	print
	print("T1 Spheres")
	print("9 sphere layer")
	angle = 360 / t1_large
	for i in range(t1_large):
		degree_angle = i*angle
		print("vial {}: x: {}, y: {}".format(i + 2, \
			t1_center[0] - np.cos(np.radians(degree_angle)) * t1_large_radius, \
			t1_center[1] + np.sin(np.radians(degree_angle)) * t1_large_radius))

	print("3 sphere layer")
	angle = 360 / t1_small
	for i in range(t1_small):
		degree_angle = i*angle + t1_small_offset
		print("vial {}: x: {}, y: {}".format(i + 1, \
			t1_center[0] - np.cos(np.radians(degree_angle)) * t1_small_radius, \
			t1_center[1] + np.sin(np.radians(degree_angle)) * t1_small_radius))


