import os
import random

folder_path = r'D:\computer_vision\DATASETS\test'

# 1. Get all files in that directory
filenames = os.listdir(folder_path)

# 2. Scramble the list order in Python's memory
random.shuffle(filenames)

# 3. Physically rename the files on your disk with a number prefix
for index, name in enumerate(filenames, start=1):
    old_path = os.path.join(folder_path, name)
    
    # Creates names like: 0001_image.jpg, 0002_photo.png
    new_name = f"{index:04d}_{name}"
    new_path = os.path.join(folder_path, new_name)
    
    os.rename(old_path, new_path)

print("Check your Windows folder now! The files are physically reordered.")
