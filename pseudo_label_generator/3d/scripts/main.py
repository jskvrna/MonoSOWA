import sys
import argparse
from main_class import MainClass

def parse_args(argv):
    parser = argparse.ArgumentParser(description='Main script for 3D object detection')
    parser.add_argument('--dataset', type=str, default='kitti', choices=['kitti', 'waymo', 'all', 'waymo_converted'], help='Dataset to use: kitti, waymo, waymo_converted or all (default: kitti)')
    parser.add_argument('--action', type=str, default='demo', choices=['lidar_scans', 'transformations', 'homographies', 'mask_tracking',
                                 'frames_aggregation', 'optimization', 'dimensions_output', 'demo', 'candidates'], help=('Action to perform: lidar_scans, transformations, homographies, '
                              'mask_tracking, frames_aggregation, optimization, dimensions_output, demo (default: demo), candidates'))
    parser.add_argument('--config', type=str, default='../configs/config.yaml', help='Path to the configuration file (default: ../configs/config.yaml)')
    parser.add_argument('--seq_start', type=int, default=-1, help='Sequence start index (default: -1 -> use the one in config)')
    parser.add_argument('--seq_end', type=int, default=-1, help='Sequence end index (default: -1 -> use the one in config)')

    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    autolabel = MainClass(args)

    if autolabel.cfg.frames_creation.tracker_for_merging != '2D' and args.action == 'homographies':
        raise ValueError('Homographies can only be generated with 2D tracker')
    
    if args.dataset == 'waymo':
        if autolabel.cfg.custom_dataset.use_custom_dataset:
            autolabel.main_custom(args)
        else:
            autolabel.main_waymo(args)
    elif args.dataset == 'kitti':
        autolabel.main_kitti(args)
    elif args.dataset == 'all':
        autolabel.main_kitti360(args)
    elif args.dataset == 'waymo_converted':
        autolabel.main_waymo_converted(args)
