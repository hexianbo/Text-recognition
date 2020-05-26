import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import math
import time

def getMatchNum(matches):
    '''返回特征点匹配数量和匹配掩码'''
    matchDis=0
    for d in matches:
        matchDis-=d.distance
    return matchDis


def store(queryPath, feature_dict, feature_store):
    orb = cv2.ORB_create(edgeThreshold=20)
    for p in os.listdir(queryPath):
        ab_path = queryPath+p
        queryImage=cv2.imread(ab_path)
        # queryImage=cv2.resize(queryImage, (8, 8))
        kp2, des2 = orb.detectAndCompute(queryImage, None) #提取比对图片的特征
        feature_dict[p] = des2
        # print(des2)
    np.savez(feature_store, **feature_dict)


def compare(path, samplePath, feature_store, from_path):
    queryPath=path #图库路径
    scoreList = []
    #创建ORB特征提取器
    orb = cv2.ORB_create(edgeThreshold=20)
    bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    sampleImage=cv2.imread(samplePath)
    kp1, des1 = orb.detectAndCompute(sampleImage, None) #提取样本图片的特征
    # print(des1)
    feature_dict = {}
    # feature_store = 'features.npz'

    """
    store 函数用于储存图像特征到features.npz文件，第二次运行时可以注释掉以下这一行
    """
    store(queryPath, feature_dict, feature_store)

    features = np.load(feature_store)
    for p in os.listdir(queryPath):
        des2 = features[p]
        matches=bf.match(des1,des2) #匹配特征点
        matchDis=getMatchNum(matches) #通过比率条件，计算出匹配程度
        scoreList.append((from_path + '/' + p, matchDis))
    scoreList.sort(key=lambda x: x[1], reverse=True)
    return scoreList
