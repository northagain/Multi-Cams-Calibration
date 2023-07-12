"""
@Project: Calibrate Cameras
@File: validation.py
@Function: validate the results.
@Date: 23/06/17
"""
import numpy as np
import cv2
from calibrate_initial import *
from match_points import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def get_pts(file):
    with open(file, 'r') as file:
        lines = file.readlines()
    # 提取匹配点坐标
    pts = []

    for line in lines:
        # 解析每一行的坐标
        x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line.strip().split(','))
        # 存储到pts1和pts2数组中
        pts.append([x1, y1])
        pts.append([x2, y2])
        pts.append([x3, y3])
        pts.append([x4, y4])
    # 转换为NumPy数组
    pts = np.array(pts)

    return pts

def get_2dto3dpoints(K1,K2,R,T,pts1,pts2):

    #获得相机1的投影矩阵
    P1 = K1@np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])
    # print(K1.shape)
    #获得相机2的投影矩阵
    # print(K2.shape)
    # print(R.shape)
    # print(T.shape)
    # print(P1)
    P2 = K2@np.hstack((R, T))
    # print(P2)
    #利用匹配点和投影矩阵，采用三角测量算法可以获得三维坐标
    points_4d_homogeneous = cv2.triangulatePoints(P1, P2, pts1.T,pts2.T)
    # print(points_4d_homogeneous)
    #转为齐次坐标
    points_3d_homogeneous = points_4d_homogeneous / points_4d_homogeneous[3]
    points_3d = points_3d_homogeneous[:3].T

    return points_3d , points_3d_homogeneous

if __name__ == '__main__':
    #待验证的相机数量
    nums_cameras=3

    ptss=[]
    #获取各相机的L型匹配点
    for i in range(nums_cameras):
        pt_cam="validation/"+str(i+1)+".txt"
        pts=get_pts(pt_cam)
        # print(pts)
        ptss.append(pts)

    #读取获得的内外参数
    Ks=np.loadtxt("Result\IntrinsicMatrix.txt",delimiter=',')
    Rs=np.loadtxt("Result\RotationMatrix.txt",delimiter=',')
    Ts=np.loadtxt("Result\Translation.txt",delimiter=',')

    ACss=[]
    ABss=[]
    DCss=[]

    points_3ds=[]

    for i in range(nums_cameras-1):
        dot_ADC=[]
        dot_BDC=[]

        points_3d,points_3d_homogeneous=get_2dto3dpoints(Ks[0].reshape(3,3),Ks[i+1].reshape(3,3),Rs[i+1].reshape(3,3),Ts[i+1].reshape(3,1),ptss[0],ptss[i+1])
        ACs=[]
        ABs=[]
        DCs=[]
        points_3ds.append(points_3d)
        # np.savetxt("pts"+str(i)+".txt",points_3d)
        ZCs=[]
        AB_weighted_averages=[]
        AC_weighted_averages=[]
        DC_weighted_averages=[]
        for j in range(0,points_3d.shape[0],4):

            #读取A,B,C,D点
            A=points_3d[j]
            B=points_3d[j+1]
            C=points_3d[j+2]
            D=points_3d[j+3]

            # 按照点C距离原点的距离进行加权平均
            ZCs.append(C[2])
        #     #对三维的A,B,C,D进行可视化
        #     x=points_3d[j:j+4,0]
        #     y=points_3d[j:j+4,1]
        #     z=points_3d[j:j+4,2]
        #     # 创建三维坐标系
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')

        #     # 绘制四个点
        #     ax.scatter(x,y,z, c=['red', 'green', 'blue', 'purple'])

        #     # 设置坐标轴标签
        #     ax.set_xlabel('X')
        #     ax.set_ylabel('Y')
        #     ax.set_zlabel('Z')

        #     # 显示图形
        #     plt.show()

            vector_DC = D - C
            vector_BC = B - C
            vector_AC = A - C

            ACs.append(np.linalg.norm(A-C))
            ABs.append(np.linalg.norm(A-B))
            DCs.append(np.linalg.norm(D-C))

            ACss.append(np.linalg.norm(A-C))
            ABss.append(np.linalg.norm(A-B))
            DCss.append(np.linalg.norm(D-C))


            # print(vector_AC,vector_AD)
            #保存内积，理论上两个向量垂直，即内积为0，归一化后即为cos角度
            dot_ADC.append(np.dot(vector_AC/np.linalg.norm(A-C), vector_DC/np.linalg.norm(D-C)))
            dot_BDC.append(np.dot(vector_BC/np.linalg.norm(B-C), vector_DC/np.linalg.norm(D-C)))
        #输出内积均值

        #C点小的值权重大
        weights = 1 / np.array(ZCs)
        AC_weighted_average = np.average(np.array(ACs), weights=weights)
        AB_weighted_average = np.average(np.array(ABs), weights=weights)
        DC_weighted_average = np.average(np.array(DCs), weights=weights)

        AC_weighted_averages.append(AC_weighted_average)
        AB_weighted_averages.append(AB_weighted_average)
        DC_weighted_averages.append(DC_weighted_average)

        # print("AC:",AC_weighted_average)
        # print("AB:",AB_weighted_average)
        # print("DC:",DC_weighted_average)
        print("第",i+1,"对相机的余弦输出及其角度：")
        print("cosADC:",np.mean(dot_ADC),"∠ADC=",np.arccos(np.mean(dot_ADC))*180/np.pi)
        print("cosBDC:",np.mean(dot_BDC),"∠BDC=",np.arccos(np.mean(dot_BDC))*180/np.pi)

        # print(np.mean(ACs))
        # print(np.mean(ABs))
        # print(np.mean(DCs))
    ACss=np.array(ACss).reshape(-1,1)
    ABss=np.array(ABss).reshape(-1,1)
    DCss=np.array(DCss).reshape(-1,1)
    np.savetxt("distance.csv",np.concatenate([ACss,ABss,DCss],axis=1),header="AC,AB,DC")
    # np.savetxt("distanceAB.csv",ABss)
    # np.savetxt("distanceDC.csv",DCss)

    print("AC_weights:",np.mean(AC_weighted_average))
    print("AB_weights:",np.mean(AB_weighted_average))
    print("DC_weights:",np.mean(DC_weighted_average))

    print("AC:",np.mean(ACss))
    print("AB:",np.mean(ABss))
    print("DC:",np.mean(DCss))


