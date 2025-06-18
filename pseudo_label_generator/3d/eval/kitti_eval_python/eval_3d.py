from evaluate import evaluate

gt_path = "/path/to/kitti/training/label_2"
gt_split_file = "../../data/val.txt" # from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz

det_path = "/path/to/output/labels_best_cropped_ref"
evaluate(gt_path,det_path,gt_split_file, score_thresh=0., current_class=[0])