"""
@Project: Calibrate Cameras
@File: calibrate_stereo.py
@Function: Calibrate all cameras.
@Date: 23/06/17
"""
import numpy as np
import cv2
from calibrate_initial import *
from match_points import *

def get_pts(file):
    with open(file, 'r') as file:
        lines = file.readlines()
    # 提取匹配点坐标
    pts = []

    for line in lines:
        # 解析每一行的坐标
        x1, y1, x2, y2, x3, y3 = map(float, line.strip().split(','))
        # 存储到pts1和pts2数组中
        pts.append([x1, y1])
        pts.append([x2, y2])
        pts.append([x3, y3])
    # 转换为NumPy数组
    pts = np.array(pts)

    return pts

def get_2dto3dpoints(K1,K2,R,T,pts1,pts2):

    #获得相机1的投影矩阵
    P1 = K1@np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])
    # print(P1)

    #获得相机2的投影矩阵
    P2 = K2@np.hstack((R, T))
    # print(P2)

    #利用匹配点和投影矩阵，采用三角测量算法可以获得三维坐标
    points_4d_homogeneous = cv2.triangulatePoints(P1, P2, pts1.T,pts2.T)

    #转为齐次坐标
    points_3d_homogeneous = points_4d_homogeneous / points_4d_homogeneous[3]
    points_3d = points_3d_homogeneous[:3].T

    # print("P1@points_3d_homogeneous")
    # print(P1@points_3d_homogeneous)
    return points_3d , points_3d_homogeneous

if __name__ == '__main__':
    
    ### Step1: 获取匹配点

    #保存一维标定杆图像的文件夹
    filename="test/"
    #待标定相机个数
    nums_cameras=3
    #图像个数
    nums_images=12
    
    get_matches(filename,nums_cameras,nums_images)

    ### Step2: 利用棋盘格标定相机各个内参矩阵和畸变参数

    Ks=[]
    Ds=[]
    for i in range(nums_cameras):
        # 各个相机棋盘图像的文件夹
        chess_folder="calibrate/"+str(i+1)
        _,K,D,_,_=calibrate_one_cameras(chess_folder)
        Ks.append(K)
        Ds.append(D)

    ### Step3: 分别标定相机的外参R,T
    ptss=[]
    
    Rs=[]
    Ts=[]

    Rs.append(np.eye(3))
    Ts.append(np.zeros((3,1)))
    
    #获取各相机的匹配点
    for i in range(nums_cameras):
        pt_cam="test/"+str(i+1)+".txt"
        pts=get_pts(pt_cam)
        ptss.append(pts)

    # 此处对各个相机与基础相机1进行外参标定，而非采用solvePnP
    for i in range(nums_cameras-1):
        # F, p= cv2.findFundamentalMat(ptss[0], ptss[i+1], method=cv2.FM_RANSAC, ransacReprojThreshold=3, confidence=0.95,maxIters=100)
        F, p = cv2.findFundamentalMat(ptss[0], ptss[i+1], method=cv2.FM_8POINT)
        # F, p = cv2.findFundamentalMat(ptss[0], ptss[i+1], method=cv2.FM_LMEDS)
        # 此处对各个相机与基础相机1进行外参标定，而非采用solvePnP
        # 利用基础矩阵F与两个内参矩阵K1,K2可以求出本质矩阵E
        E=Ks[0].T@F@Ks[i+1]
        # 利用recoverPose来分解本质矩阵E来得到旋转矩阵R和平移矩阵T
        points, R, T0, maskpose=cv2.recoverPose(E,ptss[0],ptss[i+1],Ks[0])
        # points, R, T0, maskpose=cv2.recoverPose(E,ptss[0],ptss[i+1])
        # print(R.shape)

        points_3d,points_3d_homogeneous=get_2dto3dpoints(Ks[0],Ks[i+1],R,T0,ptss[0],ptss[i+1])
        
        len_ACs=[]
        for j in range(0,points_3d.shape[0],3):
            A=points_3d[j]
            # B=points_3d[i+1]
            C=points_3d[j+2]

            # vector_DC = D - C
            # vector_BC = B - C
            vector_AC = A - C
            disance = np.linalg.norm(vector_AC)
            len_ACs.append(disance)

        #AC实际距离为500.23mm
        AC=500.23
        #尺度采用均值处理
        scales=AC/np.mean(len_ACs)

        T=T0*scales

        Rs.append(R)
        Ts.append(T)

        #乘以尺度以后再次投影到三维
        points_3d,points_3d_homogeneous=get_2dto3dpoints(Ks[0],Ks[i+1],R,T,ptss[0],ptss[i+1])

        len_ACs=[]
        for j in range(0,points_3d.shape[0],3):
            A=points_3d[j]
            C=points_3d[j+2]

            vector_AC = A - C
            disance = np.linalg.norm(vector_AC)
            #打印投影至三维的距离与实际距离的比值，理论值为1
            print(AC/disance)

    # 下面注释为之前的代码，精度不高，原因就在于缺少对T乘以尺度，但修改后精度还是不如双目匹配，给注释掉了。
    # F, p = cv2.findFundamentalMat(ptss[0], ptss[1], method=cv2.FM_8POINT)
    # E=Ks[0].T@F@Ks[1]
    # points, R, T0, maskpose=cv2.recoverPose(E,ptss[0],ptss[1],Ks[0])
    # points_3d,points_3d_homogeneous=get_2dto3dpoints(Ks[0],Ks[1],R,T0,ptss[0],ptss[1])
    
    # len_ACs=[]
    # for j in range(0,points_3d.shape[0],3):
    #     A=points_3d[j]
    #     # B=points_3d[i+1]
    #     C=points_3d[j+2]

    #     # vector_DC = D - C
    #     # vector_BC = B - C
    #     vector_AC = A - C
    #     disance = np.linalg.norm(vector_AC)
    #     len_ACs.append(disance)

    # AC=500.23
    # scales=AC/np.mean(len_ACs)

    # T=T0*scales

    # Rs.append(R)
    # Ts.append(T)

    # points_3d,points_3d_homogeneous=get_2dto3dpoints(Ks[0],Ks[1],R,T,ptss[0],ptss[1])

    # for m in range(nums_cameras-2):
    #     # 利用三维坐标和匹配点，采用PNP算法可以求解外参
    #     # flag,r,t=cv2.solvePnP(points_3d,ptss[i+2],Ks[i+2],distCoeffs=Ds[i+2])
    #     flag,r,t=cv2.solvePnP(points_3d,ptss[m+2],Ks[m+2],distCoeffs=Ds[m+2])
    #     # Rodrigues将旋转向量转换成旋转矩阵
    #     r, _ = cv2.Rodrigues(r)
    #     Rs.append(r)
    #     Ts.append(t)
        # print(r)
        # print(t)

    # 寻找基础矩阵F的三种算法 均需要匹配点
    # F, p= cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC, ransacReprojThreshold=3, confidence=0.95,maxIters=100)
    # F, p = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_8POINT)
    # F, p = cv2.findFundamentalMat(ptss[0], ptss[1], method=cv2.FM_LMEDS)
    
    # # 利用基础矩阵F与两个内参矩阵K1,K2可以求出本质矩阵E
    # E=Ks[0].T@F@Ks[1]

    # # 利用recoverPose来分解本质矩阵E来得到旋转矩阵R和平移矩阵T
    # points, R, T, maskpose=cv2.recoverPose(E,ptss[0],ptss[1],Ks[1])
    # # print(R.shape)
    # Rs.append(R)
    # Ts.append(T)

    # 获得世界坐标系即1相机下的各点的三维坐标
    # points_3d,points_3d_homogeneous=get_2dto3dpoints(Ks[0],Ks[1],R,T,ptss[0],ptss[1])

    # points = points_3d.reshape(41, 3,3)  # 将数据的形状修改为(16, 3, 3)，16个三维点

    # # 提取A、B、C点坐标
    # A = points[:, 0]  # A点坐标
    # B = points[:, 1]  # B点坐标
    # C = points[:, 2]  # C点坐标

    # # 计算两点
    # AB_distance = np.linalg.norm(B - A, axis=1)
    # BC_distance = np.linalg.norm(C - B, axis=1)
    # AC_distance = np.linalg.norm(C - A, axis=1)

    # # 打印结果
    # print("AB距离:", AB_distance)
    # print("BC距离:", BC_distance)
    # print("AC距离:", AC_distance)

    np.savetxt("Result/IntrinsicMatrix.txt",np.array(Ks).reshape(nums_cameras,-1),delimiter=",")
    np.savetxt("Result/DistortionParameters.txt",np.array(Ds).reshape(nums_cameras,-1),delimiter=',')
    np.savetxt("Result/RotationMatrix.txt",np.array(Rs).reshape(nums_cameras,-1),delimiter=',')
    np.savetxt("Result/Translation.txt",np.array(Ts).reshape(nums_cameras,-1),delimiter=',')


