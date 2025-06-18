from evaluate import evaluate

det_path = "/path/to/detection/results"
gt_path = "/path/to/kitti/training/label_2"
gt_split_file = "/path/to/validation/split/file.txt" # from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz

for i in range(100):
    evaluate(gt_path,det_path,gt_split_file, score_thresh=i/100., current_class=[0])