import cv2
import numpy as np
from copy import deepcopy
from collections import Counter
import logging

CANNY_THRESH_LOW = 50
CANNY_THRESH_HIGH = 200
TEMPLATE_MATCH_THRESH = 0.30
CRYSTAL_THRESH = 37

def get_ref_array(px_size):
    W=372
    H=240

    ref = np.zeros( (H,W), dtype=np.uint8)
    for y in (29, 76, 170, 217):
        for x in range(8 if y in (76, 170) else 7):
            cv2.circle(ref, (23+x*46, y), 15, 255, 2)

    # resize so rectangle is 12cm wide
    scale = px_size/0.03226
    ref = cv2.resize(ref, (int(round(W/scale)), int(round(H/scale))))

    # cv2.imwrite("ref.png", ref)
    return ref

def get_bubble_coordinates(px_size, template_corner, idx):
    scale = 0.06445/px_size

    tx = template_corner[0]
    ty = template_corner[1]

    # these values derived from 4493056_4493_T1_NIST_PHANTOM_NoPE/E7217/400/
    # could update with reference values from CAD design
    ox = int(round((5 + 18.2*idx)*scale ))
    oy = int(round(58*scale))
    dx = int(round(10*scale))
    dy = int(round(8*scale))

    return tx+ox, ty+oy, tx+ox+dx, ty+oy+dy

def get_temperature(dwi_array, image_coordinate_system, save_images=False):

    px_size = image_coordinate_system.pixel_spacing_cm
    slice_size = image_coordinate_system.spacing_between_slices_cm

    logging.info("Using DWI Component %d of %d" % (dwi_array.shape[3], dwi_array.shape[3]))
    dwi = deepcopy(dwi_array[:, :, :, -1])
    y = np.shape(dwi)[2]

    # Build reference image:
    ref = get_ref_array(px_size)
    H, W = ref.shape

    slices = {}
    for i in range(y):
        # Extract slice
        img = dwi[:, :, i]

        # normalize, make U8C1
        slice_max = np.max(img)
        if slice_max == 0:
            continue
        img = img * 255./slice_max
        img = np.uint8(img)

        # resize so that pixels are square
        img = cv2.resize(img, (img.shape[1], int(img.shape[0]*(slice_size/px_size))))

        #img = cv2.fastNlMeansDenoising(img, 10)

        # edge detect
        edge = cv2.Canny(img, CANNY_THRESH_LOW, CANNY_THRESH_HIGH)

        # find template
        result = cv2.matchTemplate(edge, ref, cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        # Skip if match isn't good enough
        # logging.info("template match", i, maxVal)
        if maxVal < TEMPLATE_MATCH_THRESH:
            continue

        # Extract values from each bubble
        vals = []
        for t in range(10):
            x1, y1, x2, y2 = get_bubble_coordinates(px_size, maxLoc, t)
            norm = (x2-x1)*(y2-y1)
            vals.append(np.sum(img[y1:y2, x1:x2])/norm)
            if save_images:
                # img[y1:y2, x1:x2]=192
                cv2.rectangle(img, (x1, y1), (x2, y2), 255, 1)

        slices[i] = {'maxLoc' : maxLoc, 'maxVal' : maxVal, 'therm' : np.array(vals) }

        if save_images:
            # Annotate images
            cv2.rectangle(edge, maxLoc, (maxLoc[0]+W, maxLoc[1]+H), 255, 2)
            edge[maxLoc[1]:maxLoc[1]+H,maxLoc[0]:maxLoc[0]+W] |= ref
            cv2.rectangle(img, maxLoc, (maxLoc[0]+W, maxLoc[1]+H), 255, 2)
            img[maxLoc[1]:maxLoc[1]+H,maxLoc[0]:maxLoc[0]+W] |= ref

            # Save out images
            # cv2.imwrite("slice-%03d-img.png" % i, img)
            # cv2.imwrite("slice-%03d-edge.png" % i, edge)
            img = np.hstack((img, edge))
            cv2.imwrite("slice-%03d.png" % i, img)

    if len(slices) == 0:
        return None

    # Look at maxLoc and make sure all slices are in the same spot
    # reject those that don't share the "majority opinion".
    # Then, use the median for the brightness value
    loc_count = Counter()
    for s in slices.values():
        loc_count[s['maxLoc']] += 1
    best = sorted(slices.keys(), key=lambda k: (loc_count[slices[k]['maxLoc']], slices[k]['maxVal']), reverse=True)[0]
    keep = [s for s in slices.keys() if slices[s]['maxLoc'] == slices[best]['maxLoc']]
    logging.info("Using slices %s for thermometry" % (",".join(map(str, keep))))
    val2d = np.array( [slices[k]['therm'] for k in keep] )
    val = np.median(val2d, axis=0)

    for v in range(10):
        if val[9-v] > CRYSTAL_THRESH:
            break
    return 24-v
