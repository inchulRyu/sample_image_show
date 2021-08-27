import os
import random
import cv2
import natsort
import datetime
import shutil
import numpy as np
from tqdm import tqdm
current_path = os.path.dirname(os.path.abspath(__file__))

class sample_extract:

    def __init__(self, args):

        self.args = args
        self.img_folder_path = args.imgs_path
        if self.args.visualize == True:
            self.label_folder_path = args.labels_path
        self.invalid_img_list_save = os.path.join(current_path, 'static/save/invalid_img_list')
        self.sample_num = args.sample_num

        self.file_list_tmp = os.listdir(self.img_folder_path)

        # print(len(file_list_tmp))
        # print(os.path.splitext(file_list_tmp[0])[1])
        
        print('images loading')
        self.img_file_list = [i for i in self.file_list_tmp if os.path.splitext(i)[1] in ['.jpg', '.png']]
        self.img_file_list = natsort.natsorted(self.img_file_list)
        print('completed')

        # 이미 본 이미지 리스트 불러오기
        print('filtering out images already seen')
        self.already_seen_list = []
        if os.path.exists(os.path.join(current_path, 'static/save/already_seen', args.imgs_path.replace('/', '&#47;')+'_seen.txt')):
            with open(os.path.join(current_path, 'static/save/already_seen', args.imgs_path.replace('/', '&#47;')+'_seen.txt'), 'r') as f:
                already_seen_list_tmp = f.readlines()
                self.already_seen_list = list(map(lambda x: x.replace('\n',''), already_seen_list_tmp))
        else:
            with open(os.path.join(current_path, 'static/save/already_seen', args.imgs_path.replace('/', '&#47;')+'_seen.txt'), 'w') as f:
                f.write('\n'.join(self.already_seen_list))

        # 예전에 확인했었던 이미지 지우기
        for i in self.already_seen_list:
            try:
                self.img_file_list.remove(i)
            except:
                with open(os.path.join(current_path, 'static/save/already_seen', args.imgs_path.replace('/', '&#47;')+'_seen.txt'), 'w') as f:
                    f.write('\n'.join(self.already_seen_list))
        print('completed')
        
        self.already_seen_list = []

        # 잘못된 이미지 리스트 불러오기
        print('loading invalid images save file')
        self.invalid_img_list = []
        self.invalid_file_name = args.imgs_path.replace('/', '&#47;')+'_invalid.txt'
        if os.path.exists(os.path.join(self.invalid_img_list_save, self.invalid_file_name)):
            with open(os.path.join(self.invalid_img_list_save, self.invalid_file_name), 'r') as f:
                invalid_img_list_tmp = f.readlines()
                self.invalid_img_list = list(map(lambda x: x.replace('\n',''), invalid_img_list_tmp))
        print('completed')

        self.labels_info = {}

        if self.args.visualize == True:
            # self.classes_list = ['person', 'vehicle', 'twowheeler']
            print('labels scanning...')
            if self.args.label_format == 'YOLO':
                with open(self.args.names_path) as f:
                    self.classes_list = list(map(lambda x: x.replace('\n', ''), f.readlines()))
                    num_classes = len(self.classes_list)
                if self.args.label_select:
                # 라벨파일 스캔해서 저장하기
                    label_list_temps = os.listdir(self.label_folder_path)
                    for e in tqdm(label_list_temps):
                        each_file_label = np.zeros(num_classes, dtype='int8')
                        with open(os.path.join(self.label_folder_path, e), 'r') as f:
                            labels_tmp = f.readlines()
                            labels_tmp = list(map(lambda x: x.split(), labels_tmp))
                            for l in labels_tmp:
                                each_file_label[int(l[0])] += 1
                        self.labels_info[e] = each_file_label

            # KITTI.
            elif self.args.label_format == 'KITTI':
                self.classes_list = []
                print('Searching all classes from label files...')
                label_list_temps = os.listdir(self.label_folder_path)
                # class names 불러오기.
                for e in tqdm(label_list_temps):
                    each_file_label = {}
                    with open(os.path.join(self.label_folder_path, e), 'r') as f:
                        labels_tmp = f.readlines()
                        labels_tmp = list(map(lambda x: x.split(), labels_tmp))
                        for l in labels_tmp:
                            class_name = l[0]
                            if class_name == 'DontCare':
                                continue
                            if class_name not in self.classes_list:
                                self.classes_list.append(class_name)
                            if class_name in each_file_label:
                                each_file_label[class_name] += 1
                            else:
                                each_file_label[class_name] = 1
                    self.labels_info[e] = each_file_label
            print('completed')

            self.color = {name: [
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)] for name in self.classes_list}
        else:
            self.color = {}

        self.sample_img_name = []
        self.sample_img_name_tmp = []

        # 남은 이미지 수
        self.num_remaining_img = len(self.img_file_list)

        # print(len(file_list))

    def visualize(self, image, labels, thickness=4):
        for each_label in labels:

            # try:
            if self.args.label_format == 'YOLO':
                c, x, y, w, h = list(map(float,each_label))
                c = int(c)

                img_h, img_w, _ = image.shape

                class_name = self.classes_list[c]
                x_min = int((x-w/2)*img_w)
                y_min = int((y-h/2)*img_h)
                x_max = int((x+w/2)*img_w)
                y_max = int((y+h/2)*img_h)
            elif self.args.label_format == 'KITTI':
                class_name = each_label[0]
                truncated = float(each_label[1]) # 화면 밖으로 잘린 정도. 0 ~ 1
                occluded = int(each_label[2])    # 다른 물체에 의해 가려진 정도. 0=fully visible, 1=partly occluded, 2=largely occluded, 3=unknown
                if class_name == 'DontCare':
                    continue
                if truncated >= 0.9:
                    continue
                if occluded >= 3:
                    continue

                x_min, y_min, x_max, y_max = list(map(lambda x: int(float(x)),each_label[4:8]))
            # except:
            #     print(f'label file format이 {self.args.label_format}과(와) 일치하지 않거나 label file이 잘못되었습니다.')

            # bbox 좌표가 이미지 넘어갈 시 예외처리
            # if x_min < 0:
            #     x_min = 0
            # if y_min < 0:
            #     y_min = 0
            # if x_max > img_w:
            #     x_max = img_w
            # if y_max > img_h:
            #     y_max = img_h

            # bbox
            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=self.color[class_name], thickness=4)
            # class 명
            ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)  
            image = cv2.rectangle(image, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), self.color[class_name], -1)
            image =cv2.putText(
                image,
                text=class_name,
                org=(x_min, y_min - int(0.3 * text_height)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.35,
                color=(255,255,255),
                lineType=cv2.LINE_AA,
            )
        
        return image

    def random_extract(self):
        self.sample_img_name = []
        self.sample_img_name_tmp = []

        # 랜덤으로 이미지 뽑기. 리스트에서는 제거.
        if self.args.random == True:
            if self.sample_num > len(self.img_file_list):
                self.sample_num = len(self.img_file_list)

            self.slc_labels = []
            self.slc_labels_i = []
            # 라벨 인덱스 번호 추출하기.
            for l in self.slc_labels:
                self.slc_labels_i.append(self.classes_list.index(l))

            # self.filtered_img_list = []
            # for k, v in self.labels_info.items():
            #     for label_i in self.slc_labels_i:
            #         if v[label_i] >= 1:
            #             self.filtered_img_list.append(os.path.splitext(k)[0])
            #             break
            
            sample_cnt = 0
            self.sample_img_name = []

            # 라벨 선택 기능
            if self.args.label_select:
                if len(self.slc_labels) > 0:
                    while True:
                        if len(self.img_file_list) == 0:
                            break
                        # rand_index = random.sample(range(len(self.img_file_list)), 1)
                        rand_index = random.randrange(len(self.img_file_list))
                        sample_img_tmp = self.img_file_list.pop(rand_index)
                        if self.args.label_format == 'KITTI':
                            for label in self.slc_labels:
                                if label in self.labels_info[os.path.splitext(sample_img_tmp)[0]+'.txt']:
                                    self.sample_img_name.append(sample_img_tmp)
                                    sample_cnt += 1
                                    break
                        else:
                            for label_i in self.slc_labels_i:
                                if self.labels_info[os.path.splitext(sample_img_tmp)[0]+'.txt'][label_i] >= 1:
                                    self.sample_img_name.append(sample_img_tmp)
                                    sample_cnt += 1
                                    break
                        if sample_cnt == self.sample_num:
                            break
                        # 남은 이미지 수 빼주기
                        self.num_remaining_img -= 1
                else:
                    self.sample_img_name = [self.img_file_list.pop(i) for i in random.sample(range(len(self.img_file_list)), self.sample_num)]
                    # 남은 이미지 수 빼주기
                    self.num_remaining_img -= self.sample_num
            else:
                self.sample_img_name = [self.img_file_list.pop(i) for i in random.sample(range(len(self.img_file_list)), self.sample_num)]
                # 남은 이미지 수 빼주기
                self.num_remaining_img -= self.sample_num
                    
        else:
            num_of_img_read = self.sample_num*self.args.stride
            if num_of_img_read > len(self.img_file_list):
                num_of_img_read = len(self.img_file_list)
            
            for i in range(num_of_img_read):
                img_name = self.img_file_list.pop(0)
                self.sample_img_name_tmp.append(img_name)
                if i % self.args.stride == 0:
                    self.sample_img_name.append(img_name)
            # 남은 이미지 수 빼주기
            self.num_remaining_img -= num_of_img_read


    def get_img(self, index, mode):
        # try:
        if mode == 0:
            # 이미지 읽기
            img = cv2.imread(os.path.join(self.img_folder_path, self.sample_img_name[index]), 1)

            if self.args.visualize == True:
                # 라벨 읽기
                with open(os.path.join(self.label_folder_path, self.sample_img_name[index][:-3]+'txt'), 'r') as lf:
                    labels = lf.readlines()
                    labels = list(map(lambda x: x.split(), labels))

                # bbox 그리기
                img = self.visualize(img, labels, thickness=4)

        else:
            if index >= len(self.invalid_img_list):
                img = cv2.imread('static/image/x_image.png', 1)
            else:
                img = cv2.imread(os.path.join(self.img_folder_path, self.invalid_img_list[index]), 1)

                if self.args.visualize == True:
                    # 라벨 읽기
                    with open(os.path.join(self.label_folder_path, self.invalid_img_list[index][:-3]+'txt'), 'r') as lf:
                        labels = lf.readlines()
                        labels = list(map(lambda x: x.split(), labels))

                    # bbox 그리기
                    img = self.visualize(img, labels, thickness=4)

        # except:
        #     # 오류 이미지
        #     img = cv2.imread('static/image/x_image.png', 1)

        # print(self.sample_img_name[index])
        # print(img.shape)

        _, jpeg = cv2.imencode('.jpg', img)
        jpeg = jpeg.tobytes()
            
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r\n')

    def already_seen_list_save(self):
        if self.args.random == True:
            self.already_seen_list.extend(self.sample_img_name)
        else:
            self.already_seen_list.extend(self.sample_img_name_tmp)

        with open(os.path.join(current_path, 'static/save/already_seen', self.img_folder_path.replace('/', '&#47;')+'_seen.txt'), 'a+') as f:
            f.write('\n'.join(self.already_seen_list)+'\n')
            self.already_seen_list = []

    def invalid_img_save(self, index):
        if self.sample_img_name[index] not in self.invalid_img_list:
            self.invalid_img_list.append(self.sample_img_name[index])
        with open(os.path.join(self.invalid_img_list_save, self.invalid_file_name), 'w') as f:
            f.write('\n'.join(self.invalid_img_list))

    def rm_from_invalid_img_list(self, index):
        self.invalid_img_list.pop(index)
        with open(os.path.join(self.invalid_img_list_save, self.invalid_file_name), 'w') as f:
            f.write('\n'.join(self.invalid_img_list))

    def color_change(self):
        self.color = {name: [
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)] for name in self.classes_list}

    def copy_img_to_static(self, img_data):
        img_index = int(img_data['full_img_i'])
        save_mode = int(img_data['mode'])

        if save_mode == 0:
            # shutil.copyfile(os.path.join(self.img_folder_path, self.sample_img_name[img_index]), 'static/image/full_image'+os.path.splitext(self.sample_img_name[img_index])[1])

            # 이미지 읽기
            full_img = cv2.imread(os.path.join(self.img_folder_path, self.sample_img_name[img_index]), 1)

            cv2.imwrite('static/image/full_image.png', full_img)

            if self.args.visualize == True:
                # 라벨 읽기
                with open(os.path.join(self.label_folder_path, self.sample_img_name[img_index][:-3]+'txt'), 'r') as lf:
                    labels = lf.readlines()
                    labels = list(map(lambda x: x.split(), labels))

                # bbox 그리기
                full_img = self.visualize(full_img, labels, thickness=4)

            cv2.imwrite('static/image/full_image_v.png', full_img)

        else:
            # shutil.copyfile(os.path.join(self.img_folder_path, self.invalid_img_list[img_index]), 'static/image/full_image_invalid'+os.path.splitext(self.invalid_img_list[img_index])[1])

            # 이미지 읽기
            full_img = cv2.imread(os.path.join(self.img_folder_path, self.invalid_img_list[img_index]), 1)

            cv2.imwrite('static/image/full_image_invalid.png', full_img)

            if self.args.visualize == True:
                # 라벨 읽기
                with open(os.path.join(self.label_folder_path, self.invalid_img_list[img_index][:-3]+'txt'), 'r') as lf:
                    labels = lf.readlines()
                    labels = list(map(lambda x: x.split(), labels))

                # bbox 그리기
                full_img = self.visualize(full_img, labels, thickness=4)

            cv2.imwrite('static/image/full_image_invalid_v.png', full_img)



