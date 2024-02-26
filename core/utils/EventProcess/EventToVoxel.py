import numpy as np

def events_to_voxel(events: np.ndarray, num_bins:int, height:int, width:int, pos:int=0) -> np.ndarray:
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    :param events: a [N x 4] array containing one event per row in the form: [x, y, timestamp, polarity=(+1,-1)]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param pos: filter the polarity of events
    :return voxel: [B,H,W]
    """

    assert (events.shape[1] == 4)
    assert (num_bins > 0)
    assert (width > 0)
    assert (height > 0)
    assert (pos == 0 or pos == 1 or pos == -1)

    # Inital voxel
    voxel_grid = np.zeros((num_bins, height, width), np.float32)
    voxel_grid = voxel_grid.ravel()  # stretch to one-dimensional array

    # Extract information
    ts = events[:, 2]
    xs = events[:, 0].astype(np.int64)
    ys = events[:, 1].astype(np.int64)
    pols = events[:, 3]

    # Normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = ts[-1]
    first_stamp = ts[0]
    deltaT = 0 if last_stamp == first_stamp else last_stamp - first_stamp
    ts = (num_bins - 1) * (ts - first_stamp) / deltaT
    
    # Discretize t to integer fields
    tis = ts.astype(np.int64)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    # Assign events to coordinates
    valid_indices = None
    if pos == 0:
        valid_indices = tis < num_bins
    elif pos > 0:
        valid_indices = np.logical_and(tis < num_bins, pols > 0)
    elif pos < 0:
        valid_indices = np.logical_and(tis < num_bins, pols < 0)
    np.add.at(voxel_grid,
              xs[valid_indices] + ys[valid_indices] * width + tis[valid_indices] * width * height,
              vals_left[valid_indices])
    if pos == 0:
        valid_indices = (tis + 1) < num_bins
    elif pos > 0:
        valid_indices = np.logical_and((tis + 1) < num_bins, pols > 0)
    elif pos < 0:
        valid_indices = np.logical_and((tis + 1) < num_bins, pols < 0)
    np.add.at(voxel_grid,
              xs[valid_indices] + ys[valid_indices] * width + (tis[valid_indices] + 1) * width * height,
              vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))
    return voxel_grid
