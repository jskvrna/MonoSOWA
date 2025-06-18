import os
import datetime
import kitti_eval_python.kitti_common as kitti
from kitti_eval_python.eval import get_official_eval_result

# Placeholder paths – replace with your actual directories or pass via CLI
PREDICTION_PATH = "PATH/TO/YOUR/PREDICTIONS"
GT_PATH         = "PATH/TO/YOUR/GROUND_TRUTHS"
VAL_IDXS_PATH   = "PATH/TO/YOUR/VAL_IDX_LIST.txt"

def eval(results_dir, gt_dir, val_idxs_path, logger=None):
    print("==> Loading detections and GTs...")
    with open(val_idxs_path, 'r') as f:
        idx_list = [line.strip() for line in f]
    img_ids    = [int(i) for i in idx_list]
    dt_annos   = kitti.get_label_annos(results_dir, img_ids)
    gt_annos   = kitti.get_label_annos(gt_dir, img_ids)
    test_id    = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

    print("==> Evaluating (official)...")
    car_moderate = 0.0
    for category in ['Car']:
        results_str, _, mAP3d_R40 = get_official_eval_result(
            gt_annos, dt_annos, test_id[category]
        )
        if category == 'Car':
            car_moderate = mAP3d_R40
        print(results_str)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        out_file  = os.path.join(results_dir, f"eval_{timestamp}.txt")
        with open(out_file, 'w') as f:
            f.write(results_str)

    return car_moderate

if __name__ == "__main__":
    evaluator_score = eval(
        PREDICTION_PATH,
        GT_PATH,
        VAL_IDXS_PATH
    )
    print(f"Car moderate mAP3d@R40: {evaluator_score:.4f}")
