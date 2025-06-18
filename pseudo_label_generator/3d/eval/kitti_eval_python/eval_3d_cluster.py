from evaluate import evaluate
import sys

gt_path = "/path/to/kitti/training/label_2"
gt_split_file = "../../data/val.txt" # from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz

if len(sys.argv) < 2:
    print("Please provide the folder path as an argument.")
    sys.exit(1)


det_path = sys.argv[1]
evaluate(gt_path,det_path,gt_split_file, score_thresh=0., current_class=[0])