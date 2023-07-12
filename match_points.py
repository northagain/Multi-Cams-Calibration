"""
@Project: Calibrate Cameras
@File: match_points.py
@Function: Load all the images in a 1D-calibrate pole folder to return all the matched points.
@Date: 23/05/19
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_distance(point1, point2):
    # 输入： 像素坐标1，像素坐标2
    # 输出： 欧氏距离

    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def get_matches(file,nums_cameras,nums_images): 
    # 输入： 文件路径，相机个数，图像个数
    # 输出： 保存的各相机匹配点txt文件

    #有两个阈值需要单独设置，一是二值化阈值，二是过滤面积阈值
    #对于二值化阈值，这里采用的是统计每个像素点出现的次数，统计标定杆的点像素值出现次数小于2时的值作为阈值
    #对于过滤面积阈值，实际上是获得二值化图像后，可能存在某些噪点（面积很小），需要剔除掉，这里直接采用3

    for m in range(nums_cameras):
        print("当前获取相机",m+1,"的匹配点")
        points_2d=[]
        for n in range(nums_images):

            # image_name=file+"L"+str(m+1)+"-1-"+str(n+1)+".bmp"
            image_name=file+"L"+str(m+1)+"-{:02d}.bmp".format(n + 1)

            # print(image_name)
            image = cv2.imread(image_name, 0)  # 以灰度模式读取图像
            # cv2.imshow('Detected Spots', image)
            # cv2.waitKey(100)

            # 使用unique函数获取唯一值和它们的计数
            unique_values, counts = np.unique(image.reshape(image.shape[0]*image.shape[1]), return_counts=True)
            first_occurrence = unique_values[counts < 2][0]

            # 二值化（根据需要调整阈值）
            # print(np.max(image))
            threshold_value = first_occurrence
            _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

            # 提取轮廓
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # print(contours)

            # 过滤小的轮廓（可根据需要调整阈值）
            min_contour_area = 4
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

            # cv2.imshow('Detected Spots', binary_image)
            # cv2.waitKey(100)

            # print(filtered_contours)
            # 计算光斑质心并确定位置
            spots = []
            for cnt in filtered_contours:
                # 计算轮廓的质心
                moments = cv2.moments(cnt)
                # center_x = int(moments['m10'] / moments['m00'])
                # center_y = int(moments['m01'] / moments['m00'])
                
                center_x = moments['m10'] / moments['m00']
                center_y = moments['m01'] / moments['m00']
                
                # print(center_x,center_y)
                spots.append((center_x, center_y))
            # print(len(spots))
            
            # 根据标定杆上三点共线原则剔除干扰光斑，保留标定杆上的光斑
            # 三点的筛选（可根据需要调整阈值）
            threshold_T = 600  # 阈值
            valid_spots = []
            for i in range(len(spots)):
                A = spots[i]
                for j in range(i+1, len(spots)):
                    B = spots[j]
                    AB_vector = np.array([B[0] - A[0], B[1] - A[1]])
                    for k in range(j+1, len(spots)):
                        C = spots[k]
                        AC_vector = np.array([C[0] - A[0], C[1] - A[1]])
                        cross_product = np.cross(AB_vector, AC_vector)
                        # print(np.abs(cross_product))
                        if np.abs(cross_product) < threshold_T:
                            # print(np.abs(cross_product))
                            valid_spots.extend([A, B, C])

            # cv2.destroyAllWindows()

            # 计算所有点对之间的距离
            distances = []
            for i in range(len(valid_spots)):
                for j in range(i+1, len(valid_spots)):
                    distance = calculate_distance(valid_spots[i], valid_spots[j])
                    distances.append((i, j, distance))

            # 找到距离最大和最小的点对
            distances.sort(key=lambda x: x[2])
            min_distance_pair = distances[0]
            max_distance_pair = distances[-1]

            # 确定 A、B、C 三点的对应关系
            B_index, C_index = min_distance_pair[0], min_distance_pair[1]
            if max_distance_pair[0]==B_index:
                A_index  = max_distance_pair[1]
            else:
                A_index  = max_distance_pair[0]
            # 获取对应的点坐标
            A = valid_spots[A_index]
            B = valid_spots[B_index]
            C = valid_spots[C_index]

            # print(A)
            # print(valid_spots[A_index])
            # 在原图像上绘制光斑位置
            spot_radius = 4
            spot_color1 = (0, 0, 255)  # A红 red
            spot_color2 = (0, 255, 0)  # B绿 green
            spot_color3 = (255, 0, 0)  # C蓝 blue

            # 对每张图像的ABC点进行绘制可视化
            for spot in valid_spots:
            # for spot in spots:
                image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                # cv2.circle(binary_image, (int(A[0]),int(A[1])), spot_radius, spot_color1, -1)
                # cv2.circle(binary_image, (int(B[0]),int(B[1])), spot_radius, spot_color2, -1)
                # cv2.circle(binary_image, (int(C[0]),int(C[1])), spot_radius, spot_color3, -1)
                cv2.circle(image_color, (int(A[0]),int(A[1])), spot_radius, spot_color1, -1)
                cv2.circle(image_color, (int(B[0]),int(B[1])), spot_radius, spot_color2, -1)
                cv2.circle(image_color, (int(C[0]),int(C[1])), spot_radius, spot_color3, -1)

            cv2.imshow('Detected Spots', image_color)
            cv2.waitKey(100)

            #检测异常点，一般不会出现下面的输出，但如果存在某些特殊的噪声，可能会出现。
            if A==B:
                print("A==B!!!!")
                print(image_name)
            if A==C:
                print("A==C!!!!")
                print(image_name)
            if B==C:
                print("B==C!!!!")
                print(image_name)

            points_2d.append(np.array([A, B, C]).flatten())
            # print(m,n,points_2d)
        np.savetxt(file+'/'+str(m+1)+".txt",points_2d,delimiter=',')

if __name__ == '__main__':
    
    filename="test/"
    nums_cameras=3
    nums_images=12

    get_matches(filename,nums_cameras,nums_images)