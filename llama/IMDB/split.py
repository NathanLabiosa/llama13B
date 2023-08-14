import os

# Path to the 'test' folder
test_folder = 'dataset/test'

# List all files in the 'test' folder
all_files = [f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))]

# Sort files alphabetically
all_files.sort()

# Check the number of files
num_files = len(all_files)

# If there are more than 2,000 files, select files to be removed
if num_files > 2000:
    # Select files after the 2000th one to be removed
    files_to_remove = all_files[2000:]
    
    # Delete the selected files
    for file in files_to_remove:
        os.remove(os.path.join(test_folder, file))

    print(f"Removed {len(files_to_remove)} files. 2000 files remain in the 'test' folder.")
else:
    print(f"There are already {num_files} files or fewer in the 'test' folder.")
