import os
import shutil
from tqdm import tqdm

current_path = os.path.dirname(os.path.abspath(__file__))

imgs_path = '/local_hdd/works/ICRyu_workspace/for_yolo/dataset/data2/image_2'
labels_path = '/local_hdd/works/ICRyu_workspace/for_yolo/dataset/data2/label_2_cvt'
img_save_path = '/local_hdd/works/ICRyu_workspace/for_yolo/dataset/data2/filtered_images'
label_save_path = '/local_hdd/works/ICRyu_workspace/for_yolo/dataset/data2/filtered_labels'

if os.path.exists(img_save_path):
    shutil.rmtree(img_save_path)
if os.path.exists(label_save_path):
    shutil.rmtree(label_save_path)
os.makedirs(img_save_path, exist_ok=True)
os.makedirs(label_save_path, exist_ok=True)

invalid_img_list_save = os.path.join(current_path, 'static/save/invalid_img_list')
invalid_file_name = imgs_path.replace('/', '&#47;')+'_invalid.txt'

if os.path.exists(os.path.join(invalid_img_list_save, invalid_file_name)):
    with open(os.path.join(invalid_img_list_save, invalid_file_name), 'r') as f:
        invalid_img_list_tmp = f.readlines()
        invalid_img_list = list(map(lambda x: x.replace('\n',''), invalid_img_list_tmp))

img_list = os.listdir(imgs_path)

print(f'total imgs : {len(img_list)}')
print(f'invalid imgs : {len(invalid_img_list)}')

for i in invalid_img_list:
    img_list.remove(i)

print(f'filtered imgs : {len(img_list)}')

for i in tqdm(img_list):
    file_name = os.path.splitext(i)[0]
    ext = os.path.splitext(i)[1]
    # copy image
    shutil.copyfile(os.path.join(imgs_path, i), os.path.join(img_save_path, i))
    # copy label
    shutil.copyfile(os.path.join(labels_path, file_name+'.txt'), os.path.join(label_save_path, file_name+'.txt'))