import os
import shutil
import random

# Path to extracted data
dataset_path = r"Data\Garbage_Classification"
output_path = r"dataset"  # will create train/validation here

train_ratio = 0.8  # 80% train, 20% validation

# Create train/validation folders
for split in ['train', 'validation']:
    for cls in os.listdir(dataset_path):
        os.makedirs(os.path.join(output_path, split, cls), exist_ok=True)

# Split images
for cls in os.listdir(dataset_path):
    cls_path = os.path.join(dataset_path, cls)
    imgs = [f for f in os.listdir(cls_path) if f.endswith('.jpg') or f.endswith('.png')]
    random.shuffle(imgs)
    split_idx = int(len(imgs) * train_ratio)

    for img in imgs[:split_idx]:
        shutil.copy(os.path.join(cls_path, img), os.path.join(output_path, 'train', cls, img))
    for img in imgs[split_idx:]:
        shutil.copy(os.path.join(cls_path, img), os.path.join(output_path, 'validation', cls, img))

print("Data preparation complete!")