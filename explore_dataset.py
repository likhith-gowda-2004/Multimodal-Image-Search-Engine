import os
import kagglehub

def explore_dataset():
    # Download/access the dataset
    path = kagglehub.dataset_download("aladdinpersson/flickr8kimagescaptions")
    print(f"Dataset path: {path}")
    
    # List root contents
    root_files = os.listdir(path)
    print("\nRoot directory contents:")
    for item in root_files:
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            print(f" - {item}/ (directory)")
        else:
            print(f" - {item} ({os.path.getsize(item_path)} bytes)")
    
    # Look for captions file or similar files
    print("\nSearching for captions file:")
    for root, dirs, files in os.walk(path):
        for file in files:
            if 'caption' in file.lower() or file.endswith('.txt'):
                file_path = os.path.join(root, file)
                print(f" - {os.path.relpath(file_path, path)} ({os.path.getsize(file_path)} bytes)")

if __name__ == "__main__":
    explore_dataset()