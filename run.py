import os
import cv2
from tqdm import tqdm
from flask import Flask, render_template, Response, jsonify, request
from sample_extract import sample_extract
import argparse

current_folder = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

def parse():
    parse = argparse.ArgumentParser(description='Image Dataset Sample SHOW')
    parse.add_argument('--visualize', default=False)
    
    # parse.add_argument('--imgs_path', default='/local_hdd/works/ICRyu_workspace/for_yolo/dataset/data2/image_2')
    # parse.add_argument('--labels_path', default='/local_hdd/works/ICRyu_workspace/for_yolo/dataset/data2/label_2')
    # parse.add_argument('--names_path', default='/local_hdd/works/ICRyu_workspace/for_yolo/dataset/obj_names.txt')
    # parse.add_argument('--imgs_path', default='/local_hdd/works/ICRyu_workspace/for_yolo/dataset/novacos_filtered')
    # parse.add_argument('--labels_path', default='/local_hdd/works/ICRyu_workspace/for_yolo/dataset/novacos_filtered')
    # parse.add_argument('--names_path', default=os.path.join(current_folder, 'obj_names.txt'))
    parse.add_argument('--imgs_path', default='/DL_data/Spocado/data/yolo_train_dataset_v04')
    parse.add_argument('--labels_path', default='/DL_data/Spocado/data/yolo_train_dataset_v04')
    parse.add_argument('--names_path', default='/local_hdd/works/ICRyu_workspace/for_yolo/train/spocado/yolov4_spocado_v2.names')
    # parse.add_argument('--imgs_path', default='/DL_data_big/mobility_aids/Images_RGB')

    parse.add_argument('--sample_num', type=int, default=12)
    parse.add_argument('--label_format', default='YOLO') # KITTI, YOLO
    parse.add_argument('--label_select', default=False, help='function that can see selected label')
    # parse.add_argument('--invalid_imgList_path', default='/DL_data/object_detection_dataset/yolo/aichallenge')

    parse.add_argument('--random', default=False)
    parse.add_argument('--stride', type=int, default=1)

    args = parse.parse_args()

    return args

@app.route('/')
def index():
    # img_f_name_list = sp_img_extr.random_extract()
    img_f_name_list = sp_img_extr.sample_img_name
    color_dict = sp_img_extr.color
    num_remaining_img = sp_img_extr.num_remaining_img
    return render_template('main.html', img_name_list=img_f_name_list, color_dict=color_dict, num_remaining_img=num_remaining_img)

@app.route('/img_show/<i><mode>')
def img_show(i,mode):
    # mode => 0(sample images show), 1(invalid images show)
    return Response(sp_img_extr.get_img(int(i),int(mode)), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/invalid_img', methods=['POST'])
def invalid_img():
    global invalid_img_index
    data = request.get_json()
    invalid_img_index = int(data['invalid_img_i'])

    sp_img_extr.invalid_img_save(invalid_img_index)
    return jsonify(result='success')

@app.route('/rm_from_invalid_list', methods=['POST'])
def rm_from_invalid_list():
    data = request.get_json()
    rm_index = int(data['rm_img_i'])

    sp_img_extr.rm_from_invalid_img_list(rm_index)
    return jsonify(result='success')

@app.route('/show_invalid_images')
def show_invalid_images():
    invalid_imgs_list = sp_img_extr.invalid_img_list
    return render_template('invalid_imgs.html', img_name_list=invalid_imgs_list)

@app.route('/already_seen_img_save', methods=['POST'])
def already_seen_img_save():
    sp_img_extr.already_seen_list_save()
    sp_img_extr.random_extract()
    return jsonify(result='success')

@app.route('/color_change', methods=['POST'])
def color_change():
    sp_img_extr.color_change()
    return jsonify(result='success')

@app.route('/copy_full_img', methods=['POST'])
def copy_full_img():
    full_img_data = request.get_json()
    sp_img_extr.copy_img_to_static(full_img_data)
    return jsonify(result='success')

@app.route('/class_filtering', methods=['POST'])

def main():
    global sp_img_extr

    args = parse()
    sp_img_extr = sample_extract(args)

    sp_img_extr.random_extract()
    app.run(host='0.0.0.0')

if __name__ == "__main__":
    main()