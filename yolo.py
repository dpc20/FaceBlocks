import os
import time
import math
import csv
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm
from collections import deque
from itertools import islice
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import random_split
from torchvision.utils import make_grid
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
import torch.nn.functional as F
from ultralytics import YOLO

"""
定义了一个名为’keypointDetect’的函数，该函数接收模型和图像作为输入，并输出关键点列表。这个函数很可能是用于检测图像中的人脸关键点。
定义了一个名为’chinPredict’的函数，该函数接收关键点列表和图像作为输入，并输出扩展后的关键点列表。这个函数似乎用于预测并添加下巴的对称点。
定义了一个名为’drawKeypoints’的函数，该函数接收文件路径和关键点列表作为输入，并在图像上绘制关键点。这个函数用于可视化关键点。
定义了一个名为’headPoseEstimate’的函数，该函数接收图像和关键点列表作为输入，并输出估计的头部姿态和图像。这个函数用于估计头部姿势。
定义了一个名为’signalGenerate’的函数，该函数接收旋转向量、关键点列表和图像作为输入，并输出信号。这个函数用于根据头部姿势和关键点生成信号。
定义了一个名为’getSignal’的函数，该函数接收模型、关键点数量、图像和记录标志作为输入，并输出关键点列表、旋转向量、图像和信号。这个函数似乎是整个过程的封装，用于获取和处理信号。
定义了一个名为’process_frame’的函数，该函数接收图像作为输入，并输出信号和图像。这个函数似乎是用于处理视频帧，检测关键点，估计头部姿势，生成信号，并将处理后的帧保存到文件中。
最后，代码检查是否存在名为’result’的文件夹，如果不存在则创建它。这个文件夹用于存储处理结果，包括原始图像、关键点坐标数据、旋转向量数据和信号数据。
"""

model_path = 's_best_full_50.pt'
signal_dict = {0: 'X', 1: 'U', 2: 'D', 3: 'L', 4: 'R'} # 对应信号无操作、上、下、左、右
num_keypoints = 5
if torch.cuda.is_available(): # 如果有可用的cuda设备
    device = 'cuda'  # 则设置为cuda
else: # 如果没有
    device = 'cpu'  # 则为cpu
if device == 'cpu':
    model = YOLO(model_path)
    model = model.to(device)
else:
    pass # 如果设备为cpu
def keypointDetect(model, img):
    # 利用yolo模型
    # 载入yolo模型 
    # 导入yolo需要的数据格式
    # 获得对应的关键点输出
    height, width = img.shape[:2]
    img_path = 'temp_image.jpg'
    cv2.imwrite(img_path, img)
    keypoints_list = []
    results = model(img_path)
    bboxes_keypoints = results[0].keypoints.xy.cpu().numpy().astype('uint32')
    bboxes_keypoint = bboxes_keypoints[0]
    if len(bboxes_keypoint) == 0:
        bboxes_keypoint = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    # print(bboxes_keypoint)
    keypoints_list.append((bboxes_keypoint[0][0] / width, bboxes_keypoint[0][1] / height))
    keypoints_list.append((bboxes_keypoint[1][0] / width, bboxes_keypoint[1][1] / height))    
    keypoints_list.append((bboxes_keypoint[2][0] / width,bboxes_keypoint[2][1] / height))
    keypoints_list.append((bboxes_keypoint[3][0] / width, bboxes_keypoint[3][1] / height))  
    keypoints_list.append((bboxes_keypoint[4][0] / width, bboxes_keypoint[4][1] / height))   
    # keypoints_list.append((bboxes_keypoint[2][0] / width,bboxes_keypoint[2][1] / height))    
    # keypoints_list.append((bboxes_keypoint[1][0] / width, bboxes_keypoint[1][1] / height))    
    # keypoints_list.append((bboxes_keypoint[0][0] / width, bboxes_keypoint[0][1] / height))
    # keypoints_list.append((bboxes_keypoint[4][0] / width, bboxes_keypoint[4][1] / height))   
    # keypoints_list.append((bboxes_keypoint[3][0] / width, bboxes_keypoint[3][1] / height))  
    # print(keypoints_list) 
    return keypoints_list


def chinPredict(keypoint, img):
    x1, y1 = keypoint[0]  # 获取第一个关键点的坐标
    x2, y2 = keypoint[3]  # 获取第二个关键点的坐标
    x3, y3 = keypoint[4]  # 获取第三个关键点的坐标
    h, w = img.shape[:2]
    
    y1 = 1 - y1
    y2 = 1 - y2
    y3 = 1 - y3
    
    x1 = int(w*x1)
    x2 = int(w*x2)
    x3 = int(w*x3)
    
    y1 = int(h*y1)
    y2 = int(h*y2)
    y3 = int(h*y3)
    # 计算由第二个和第三个点形成的直线的斜率
    if x3 != x2:
        slope = (y3 - y2) / (x3 - x2) + 1e-6  # 计算斜率
        intercept = y2 - slope * x2  # 计算截距
        
        # 计算垂直线的斜率
        perp_slope = -1 / slope
        
        # 计算通过第一个点的垂直线的截距
        perp_intercept = y1 - perp_slope * x1
        
        # 计算两条直线的交点
        x_intersect = (perp_intercept - intercept) / (slope - perp_slope)
        y_intersect = slope * x_intersect + intercept
        
        # 计算对称点
        x_sym = 2 * x_intersect - x1
        y_sym = 2 * y_intersect - y1
    else:
        # 如果直线是垂直的
        x_sym = 2 * x2 - x1
        y_sym = y1
    
    # print(x1, y1)
    # print(x2, y2)   
    # print(x3, y3)
    # print(x_sym, y_sym)
    # print((y3 - y2) / (x3 - x2))
    # print(-1 / ((y3 - y2) / (x3 - x2)))
    # print((y_sym - y1) / (x_sym - x1))
    x_sym = x_sym / w
    y_sym = y_sym / h
    y_sym = 1 - y_sym
    keypoint.append((x_sym, y_sym))  # 将对称点添加到关键点列表中
    return keypoint  # 返回对称点的坐标


def drawKeypoints(file_path, pred):
    import cv2
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt

    my_image = cv2.imread(file_path)
    h, w = my_image.shape[:2]
    point_size = 10
    my_image = Image.open(file_path)
    draw = ImageDraw.Draw(my_image)

    for x, y in pred:
        x = int(x * w)
        y = int(y * h)
        draw.ellipse((x - point_size, y - point_size, x + point_size, y + point_size), fill='red')
        # 显示图片
    plt.imshow(my_image)
    plt.show()
    return


def headPoseEstimate(img, keypoint, show=True):
    # 将传入的图像赋值给变量im
    im = img 
    # 获取图像的高度和宽度
    height, width = im.shape[:2]
    # 将关键点坐标按照图像的宽度和高度进行缩放，转换为图像坐标
    keypoint = [(x*width, y*height) for x, y in keypoint] 
    # 获取图像的尺寸
    size = im.shape

    # 定义2D图像上的关键点坐标，这些坐标与传入的关键点参数相对应
    image_points = np.array([
        keypoint[0],  # 鼻尖
        keypoint[5],  # 下巴
        keypoint[2],  # 左眼左角
        keypoint[1],  # 右眼右角
        keypoint[4],  # 左嘴角
        keypoint[3]   # 右嘴角
    ], dtype="double")
    
    # 定义3D模型上的关键点坐标，这些坐标是标准人脸模型的关键点
    model_points = np.array([
        (0.0, 0.0, 0.0),          # 鼻尖
        (0.0, -330.0, -65.0),     # 下巴
        (-225.0, 170.0, -135.0),  # 左眼左角
        (225.0, 170.0, -135.0),   # 右眼右角
        (-150.0, -150.0, -125.0), # 左嘴角
        (150.0, -150.0, -125.0)   # 右嘴角
    ])

    # 定义相机内参矩阵
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    # 假设没有镜头畸变，定义畸变系数矩阵
    dist_coeffs = np.zeros((4, 1))  
    # 使用solvePnP函数求解旋转和平移矩阵
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    # 将旋转向量转换为旋转矩阵
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    # 将旋转矩阵和平移向量合并为投影矩阵
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    # 从投影矩阵中提取欧拉角
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
  
    # 将欧拉角转换为度数
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    
    # 将旋转角度存储在rotation_vector数组中
    rotation_vector = np.array([roll, pitch, yaw])
    
    # 如果show为True，则在图像上绘制结果
    if show:
        # 定义坐标轴长度
        axis_length = 500
        # 定义坐标轴向量
        axes = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
        # 计算坐标轴向量在图像上的终点
        (axes_end_points, jacobian) = cv2.projectPoints(axes, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        
        # 定义鼻尖坐标
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        # 定义颜色映射
        colors = [(0, 255, 0),(255, 0, 0),(0, 0, 255)]
        # 绘制坐标轴
        for p, color in zip(axes_end_points, colors):
            p2 = (int(p[0][0]), int(p[0][1]))
            im = cv2.line(im, p1, p2, color, 3)  # 绘制线条
        
        # 定义要显示的文字
        words = ['roll: ', 'pitch: ', 'yaw: ']
        # 在图像上显示旋转角度
        # 输出对应的文字
        for j in range(len(rotation_vector)):
            im = cv2.putText(im, words[j] + ('{:05.2f}').format(float(rotation_vector[j])), (10, 30 + (50 * j)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
        
        kpt_color_map = {
        0:{'name':'nose', 'color':[0,0,255], 'radius':10},
        1:{'name':'right_eye', 'color':[255,0,0], 'radius':10},
        2:{'name':'left_eye', 'color':[0,255,0], 'radius':10},
        3:{'name':'right_mouth', 'color':[255,255,0], 'radius':10},
        4:{'name':'left_mouth', 'color':[0,255,255], 'radius':10},
        5:{'name':'chin', 'color':[0,0,255], 'radius':10}
        }
        for r, p in enumerate(keypoint):
            im = cv2.circle(im, (int(p[0]), int(p[1])), 10, tuple(kpt_color_map[r]["color"]), -1)
        # Display image in Jupyter Notebook
        # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        # plt.axis('on')
        # plt.show()
        # for w, p in zip(words, rotation_vector):
        #     print(w, p)
            
    return im, rotation_vector

def signalGenerate(rotation_vector, keypoint, img):
    rotation_vector = rotation_vector.flatten()  # 将旋转向量展平
    yaw   = rotation_vector[2]  # 绕y轴旋转的角度
    pitch = rotation_vector[1]  # 绕x轴旋转的角度
    roll  = rotation_vector[0]  # 绕z轴旋转的角度
    signal = 0  # 初始化信号
    
    h, w = img.shape[:2]
    left_eye_x, left_eye_y = keypoint[2][0], keypoint[2][1]
    right_eye_x, right_eye_y = keypoint[1][0], keypoint[1][1]
    left_eye_x = int(left_eye_x * w)
    left_eye_y = int((1 - left_eye_y) * h)
    right_eye_x = int(right_eye_x * w)
    right_eye_y = int((1 - right_eye_y) * h)
    
    threshold_left_right = 25
    threshold_left_right = math.tan(math.radians(threshold_left_right))
    eye_slope = (right_eye_y - left_eye_y) / (right_eye_x - left_eye_x + 1e-6)
    if eye_slope <  - threshold_left_right: 
        signal = 4
    elif eye_slope > threshold_left_right:
        signal = 3
    
    if signal:
        pass
    else:
        if pitch > 10:
            signal = 1
        elif pitch < -5:
            signal = 2

    print(f'眼睛斜率: {eye_slope}')
    # print('信号: ', signal)  # 打印信号
    return signal  # 返回信号

def getSignal(model, num_keypoints, img, record):
    keypoint = keypointDetect(model, img)  # 检测关键点
    keypoint = [(max(0, min(1, x)), max(0, min(1, y))) for x, y in keypoint]
    keypoint = chinPredict(keypoint, img)  # 预测下巴的对称点
    keypoint = [(max(0, min(1, x)), max(0, min(1, y))) for x, y in keypoint]
    # drawKeypoints(image_path, keypoint)
    img, rotation_vector = headPoseEstimate(img, keypoint, show=record)  # 估计头部姿态
    signal = signalGenerate(rotation_vector, keypoint, img)  # 生成信号
    return keypoint, rotation_vector, img, signal  # 返回信号

def process_frame(img):
    record = False
    img = cv2.flip(img, 1)
    keypoint, rotation_vector, img, signal = getSignal(model,  num_keypoints, img, record)
    signal_chinese = signal_dict[signal]
    start_time = time.time()
    if record:
        # 先将接收到的原始图片保存到raw_images文件夹（如果不存在则创建）
        if not os.path.exists('result'):
            os.makedirs('result')
        if not os.path.exists('result/raw_images'):
            os.makedirs('result/raw_images')
        frame_index = int(time.time() * 1000)  # 简单用时间戳作为序号示例，可修改为更合理的计数方式
        cv2.imwrite(f'result/raw_images/frame_{frame_index}.jpg', img)

        if not os.path.exists('result/landmarks_data'):
            os.mkdir('result/landmarks_data')

        # 保存本帧的关键点坐标数据，以帧的序号命名文件（这里简单示例，可以根据实际情况调整命名规则）
        csv_file_path = f'result/landmarks_data/frame_{frame_index}_landmarks.csv'
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # 写入表头，这里假设是x,y,z坐标
            csv_writer.writerow(['x', 'y'])
            for k in keypoint:
                # 提取landmark对象中的坐标信息组成列表后再写入CSV文件
                csv_writer.writerow([k[0], k[1]])

        if not os.path.exists('result/rotations_data'):
            os.mkdir('result/rotations_data')

        # 保存本帧的关键点坐标数据，以帧的序号命名文件（这里简单示例，可以根据实际情况调整命名规则）
        csv_file_path = f'result/rotations_data/frame_{frame_index}_landmarks.csv'
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # 写入表头，这里假设是x,y,z坐标
            csv_writer.writerow(['roll', 'pitch', 'yaw', 'signal'])
            k = rotation_vector
                # 提取landmark对象中的坐标信息组成列表后再写入CSV文件
            csv_writer.writerow([k[0], k[1], k[2], signal_chinese])

    # if record:
    scaler = 1
    end_time = time.time()
    epsilon = 1e-6
    FPS = 1 / (end_time - start_time + epsilon)
    text = f'FPS: {str(int(FPS))}\nSignal: {signal_chinese}'
    (img_height, img_width) = img.shape[:2]
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, 2)
    text_x = img_width - text_size[0] - 25 * scaler
    text_y = 50 * scaler
    img = cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 125, 125), 2)

    words = ['roll: ', 'pitch: ', 'yaw: ']
    # 输出对应的文字
    for j in range(len(rotation_vector)):
        img = cv2.putText(img, words[j] + ('{:05.2f}').format(float(rotation_vector[j])), (10, 30 + (50 * j)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
    
    h, w = img.shape[:2]
    kpt_color_map = {
    0:{'name':'nose', 'color':[0,0,255], 'radius':10},
    1:{'name':'right_eye', 'color':[255,0,0], 'radius':10},
    2:{'name':'left_eye', 'color':[0,255,0], 'radius':10},
    3:{'name':'right_mouth', 'color':[255,255,0], 'radius':10},
    4:{'name':'left_mouth', 'color':[0,255,255], 'radius':10},
    5:{'name':'chin', 'color':[0,0,255], 'radius':10}
    }
    for r, p in enumerate(keypoint):
        img = cv2.circle(img, (int(p[0]*w), int(p[1]*h)), 10, tuple(kpt_color_map[r]["color"]), -1)
        
    if record:
        if not os.path.exists('result/frame_images'):
            os.mkdir('result/frame_images')
        cv2.imwrite(f'result/frame_images/frame_{frame_index}.jpg', img)
    
    img = cv2.flip(img, 1)
    return signal_chinese, img

