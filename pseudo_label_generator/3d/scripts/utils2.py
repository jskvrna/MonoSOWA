import numpy as np
import zstd

#Lot of those are from pykitti

def load_velo_scan(file):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))

def load_waymoc_scan(file):
    """Load and parse a velodyne binary file."""
    scan = np.load(file)
    scan = scan['arr_0']
    return scan[:, :4]

def load_pseudo_lidar(file):
    """Load and parse a pseudo_lidar binary file."""
    loaded = np.load(file)
    pseudo_lidar = loaded['array1']
    pseudo_lidar = np.pad(pseudo_lidar, ((0, 0), (0, 1)), mode='constant', constant_values=1)
    return pseudo_lidar

def get_perfect_scale(template, waymo=False, height_kittiavg = 1.6, width_kittiavg = 2.0, length_kittiavg = 4.7):
    x_max = np.amax(template[:, 0])
    x_min = np.amin(template[:, 0])
    y_max = np.amax(template[:, 1])
    y_min = np.amin(template[:, 1])
    z_max = np.amax(template[:, 2])
    z_min = np.amin(template[:, 2])

    if waymo:
        height = z_max-z_min
        width = y_max-y_min
        length = x_max-x_min
    else:
        height = y_max - y_min
        width = x_max - x_min
        length = z_max - z_min

    #GOLF Mk5
    #height_kittiavg = 1.52608343
    #width_kittiavg = 1.78
    #length_kittiavg = 4.25

    scale_heigth = height_kittiavg/height
    scale_width = width_kittiavg/width
    scale_length = length_kittiavg/length
    if waymo:
        best_scale = [scale_length, scale_width, scale_heigth]
    else:
        best_scale = [scale_width, scale_heigth, scale_length]

    return best_scale
