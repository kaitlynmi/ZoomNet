import os
import random
import shutil

def main(dataset_path):
    # Dataset path
    images_folder = os.path.join(dataset_path, 'images/train')
    masks_folder = os.path.join(dataset_path, 'masks/train')
    
    # Output path
    output_path = "/home/miqing/data/thyroid/train13000with1000negAndothers"
    sets_division = {'train': 1600, 'val': 400, 'test': 400}

    # Get list of all image files
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Prepare a set to keep track of already selected files
    already_selected = set()

    for _set, num in sets_division.items():
        print(f"Building {_set} dictionary...")
        
        output_images_path = os.path.join(output_path, _set, 'images')
        output_masks_path = os.path.join(output_path, _set, 'masks')
        
        # Create output folders if they don't exist
        os.makedirs(output_images_path, exist_ok=True)
        os.makedirs(output_masks_path, exist_ok=True)
        
        # Select non-overlapping random samples
        selected_files = []
        while len(selected_files) < num:
            file = random.choice(image_files)
            if file not in already_selected:
                already_selected.add(file)
                selected_files.append(file)

        # Copy selected files to output paths
        count = 1
        for file in selected_files:
            # Copy image
            shutil.copy(os.path.join(images_folder, file), os.path.join(output_images_path, file))
            print(f"Saved ({count}/{num}) {os.path.join(output_images_path, file)}")

            # Check for corresponding mask files
            mask_file_with_suffix = file.replace('.jpg', '_mask.jpg') 
            mask_file_without_suffix = file  # Use the image filename directly
            
            mask_copied = False
            # Try to copy the mask with the suffix first
            if os.path.exists(os.path.join(masks_folder, mask_file_with_suffix)):
                shutil.copy(os.path.join(masks_folder, mask_file_with_suffix), os.path.join(output_masks_path, file))
                mask_copied = True
            # If not found, try the filename without the suffix
            elif os.path.exists(os.path.join(masks_folder, mask_file_without_suffix)):
                shutil.copy(os.path.join(masks_folder, mask_file_without_suffix), os.path.join(output_masks_path, file))
                mask_copied = True
            # Optionally log if a mask was not found
            if not mask_copied:
                print(f"Warning: No mask found for {file}")
            print(f"Saved ({count}/{num}) {os.path.join(output_masks_path, file)}")
            count+=1

if __name__ == '__main__':
    main('/home/zhangys/thyroidNodule_segmentation/data/Thyroid/train13000with1000negAndothers')
