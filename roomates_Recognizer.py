import cv2
import time
import numpy as np


def visualize(input, faces, fps, name, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print(
                'Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(
                    idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv2.rectangle(input, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0),
                          thickness)
            cv2.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv2.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv2.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv2.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv2.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv2.putText(input, 'FPS: {0:.2f},Roomate:{1}'.format(fps, name), (1, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# 计时器
tm = cv2.TickMeter()

# 定义输入和变量
rm1 = cv2.imread('roomates/ggsq.png')
rm2 = cv2.imread('roomates/tyy.png')
rm3 = cv2.imread('roomates/zxx.png')
rm4 = cv2.imread('roomates/hyf.png')

in_rm = cv2.imread('NoPeople.png')

new_shape = (300, 300)  # 统一缩放为 300*300
cos_thresh = 0.363  # cos阈值，距离越大越接近
L2_thresh = 1.128  # L2阈值，距离越小越接近
rm1 = cv2.resize(rm1, new_shape)
rm2 = cv2.resize(rm2, new_shape)
rm3 = cv2.resize(rm3, new_shape)
rm4 = cv2.resize(rm4, new_shape)

in_rm = cv2.resize(in_rm, new_shape)

# 初始化模型：
faceDetector = cv2.FaceDetectorYN.create('face_detection_yunet_2022mar.onnx', '', new_shape)
faceRecognizer = cv2.FaceRecognizerSF.create('face_recognizer_fast.onnx', '')

tm.start()

# 检测、对齐、提取特征：
# detect输出的是一个二维元祖，其中第二维是一个二维数组: n*15,n为人脸数，
# 15为人脸的xywh和5个关键点（右眼瞳孔、左眼、鼻尖、右嘴角、左嘴角）的xy坐标及置信度
faces1 = faceDetector.detect(rm1)
aligned_face1 = faceRecognizer.alignCrop(rm1, faces1[1][0])  # 对齐后的图片
feature1 = faceRecognizer.feature(aligned_face1)  # 128维特征

faces2 = faceDetector.detect(rm2)
aligned_face2 = faceRecognizer.alignCrop(rm2, faces2[1][0])
feature2 = faceRecognizer.feature(aligned_face2)

faces3 = faceDetector.detect(rm3)
aligned_face3 = faceRecognizer.alignCrop(rm3, faces3[1][0])
feature3 = faceRecognizer.feature(aligned_face3)

faces4 = faceDetector.detect(rm4)
aligned_face4 = faceRecognizer.alignCrop(rm4, faces4[1][0])
feature4 = faceRecognizer.feature(aligned_face4)

in_faces = faceDetector.detect(in_rm)
assert in_faces[1] is not None, 'Cannot find a face in input picture'
in_aligned_face = faceRecognizer.alignCrop(in_rm, in_faces[1][0])
in_feature = faceRecognizer.feature(in_aligned_face);

tm.stop()


# 人脸匹配值打分：
cos_score1 = faceRecognizer.match(feature1, in_feature, 0)
cos_score2 = faceRecognizer.match(feature2, in_feature, 0)
cos_score3 = faceRecognizer.match(feature3, in_feature, 0)
cos_score4 = faceRecognizer.match(feature4, in_feature, 0)

# 得分列表索引对应室友名字索引
rmlist = ['ggsq', 'tyy', 'zxx', 'hyf']
score_list = [cos_score1, cos_score2, cos_score3, cos_score4]
# 输出结果：
print('cos_score_list: ', score_list)
in_score = max(score_list)
# 如果当前最有可能的室友得分大于识别阈值，则判断该图片存在室友,注意只能识别一个室友
if in_score > cos_thresh:
    rm_dect = rmlist[score_list.index(max(score_list))]
    print('识别到室友: ', rm_dect)
    visualize(in_rm, in_faces, tm.getFPS(),rm_dect)
    cv2.imshow('in_roomates', in_rm)
else:
    print('当前图片未识别到室友')
    visualize(in_rm, in_faces, tm.getFPS(),"None")
    cv2.imshow('in_roomates', in_rm)

cv2.waitKey(0)
