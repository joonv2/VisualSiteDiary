import os
import sys
import json

if __name__ == '__main__':
    image_folder_path = sys.argv[1]
    assert os.path.isdir(image_folder_path) and os.path.exists(image_folder_path), \
        print(f'Fail to find {image_folder_path}: not exist or not a directory')
    image_files = os.listdir(image_folder_path)
    coco_dict = {'images': [], 'annotations': []}
    idx = 0
    for file in image_files:
        if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            continue
        coco_dict['images'].append({'file_name': file, 'id': idx+1})
        coco_dict['annotations'].append({"id": idx, "image_id": idx+1, "caption": "fake"})
        idx += 1

    with open(f'{image_folder_path}_val.json', 'w+') as f:
        f.write(json.dumps(coco_dict))
    print(f'Json file is successfully saved as {image_folder_path}.json')
