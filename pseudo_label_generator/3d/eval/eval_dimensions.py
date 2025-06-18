import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import os

from skimage.filters.rank import median
from tqdm import tqdm  # For progress bar


# Function to parse KITTI label files and filter for cars
def parse_kitti_label(file_path):
    labels = []
    if not os.path.exists(file_path):
        return labels  # Return empty list if file does not exist
    with open(file_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            object_type = data[0]
            # Filter for 'Car' objects only
            if object_type == 'Car':
                label = {
                    'type': object_type,
                    'x': float(data[11]),
                    'y': float(data[12]),
                    'z': float(data[13]),
                    'length': float(data[10]),
                    'width': float(data[9]),
                    'height': float(data[8])
                }
                labels.append(label)
    return labels

# Paths to the label folders (replace with your actual paths)
gt_label_folder = '/path/to/ground_truth/labels/'  # Ground truth labels
pred_label_folder = '/path/to/predicted/labels/'  # Predicted labels

# Initialize lists to collect errors over all frames
length_errors = []
width_errors = []
height_errors = []

# Total number of frames (assuming frames from 000000.txt to 007480.txt)
num_frames = 7481

for i in tqdm(range(num_frames), desc='Processing frames'):
    frame_id = '{:06d}'.format(i)
    gt_file = os.path.join(gt_label_folder, f'{frame_id}.txt')
    pred_file = os.path.join(pred_label_folder, f'{frame_id}.txt')

    # Parse labels
    pred_labels = parse_kitti_label(pred_file)
    if len(pred_labels) == 0:
        continue
    gt_labels = parse_kitti_label(gt_file)
    if len(gt_labels) == 0:
        continue

    # Mark all GT cars as unmatched initially
    for gt_obj in gt_labels:
        gt_obj['matched'] = False

    # Set distance threshold
    distance_threshold = 5.0  # meters

    # For each predicted car, find the nearest unmatched GT car within the threshold
    for pred_obj in pred_labels:
        min_distance = distance_threshold
        matched_gt_idx = None
        for idx, gt_obj in enumerate(gt_labels):
            if not gt_obj['matched']:
                distance = np.sqrt(
                    (gt_obj['x'] - pred_obj['x']) ** 2 +
                    (gt_obj['y'] - pred_obj['y']) ** 2 +
                    (gt_obj['z'] - pred_obj['z']) ** 2
                )
                if distance < min_distance:
                    min_distance = distance
                    matched_gt_idx = idx
        # If a match is found within the threshold
        if matched_gt_idx is not None:
            gt_obj = gt_labels[matched_gt_idx]
            gt_obj['matched'] = True  # Mark GT car as matched
            # Compute dimension errors
            length_errors.append(pred_obj['length'] - gt_obj['length'])
            width_errors.append(pred_obj['width'] - gt_obj['width'])
            height_errors.append(pred_obj['height'] - gt_obj['height'])
        else:
            # No match found within threshold; consider as false positive or skip
            continue

# Determine the minimum and maximum errors
min_error = min(np.min(length_errors), np.min(width_errors), np.min(height_errors))
max_error = max(np.max(length_errors), np.max(width_errors), np.max(height_errors))

# Define the bin edges centered around 0.0
bin_width = 0.1  # Adjust bin width as needed
bin_edges = np.arange(min_error - bin_width, max_error + bin_width, bin_width)

# Plot histograms centered around zero
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(length_errors, bins=bin_edges, color='blue', alpha=0.7)
plt.title('Length Error Histogram (Cars Only)')
plt.xlabel('Error (m)')
plt.ylabel('Frequency')
plt.xlim(min_error, max_error)

plt.subplot(1, 3, 2)
plt.hist(width_errors, bins=bin_edges, color='green', alpha=0.7)
plt.title('Width Error Histogram (Cars Only)')
plt.xlabel('Error (m)')
plt.xlim(min_error, max_error)

plt.subplot(1, 3, 3)
plt.hist(height_errors, bins=bin_edges, color='red', alpha=0.7)
plt.title('Height Error Histogram (Cars Only)')
plt.xlabel('Error (m)')
plt.xlim(min_error, max_error)

plt.tight_layout()
plt.show()

# Calculate the mean squared root errors
mse_length = np.mean(np.sqrt(np.power(length_errors,2)))
print(f'Mean Squared Root Length Error: {mse_length:.4f} meters')
mse_width = np.mean(np.sqrt(np.power(width_errors,2)))
print(f'Mean Squared Root Width Error: {mse_width:.4f} meters')
mse_height = np.mean(np.sqrt(np.power(height_errors,2)))
print(f'Mean Squared Root Height Error: {mse_height:.4f} meters')
median_length = np.median(np.sqrt(np.power(length_errors,2)))
print(f'Median Length Error: {median_length:.4f} meters')
median_width = np.median(np.sqrt(np.power(width_errors,2)))
print(f'Median Width Error: {median_width:.4f} meters')
median_height = np.median(np.sqrt(np.power(height_errors,2)))
print(f'Median Height Error: {median_height:.4f} meters')
