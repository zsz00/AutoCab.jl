
# 相机测试, 相机标定(标定板法), 图片去畸变
# python3.x + opencv3.4   2018.10.12
import os, time, math
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
# import metrics


def normal_camera_calibrate():
    # 普通(广角)相机标定. 通过标定板的标定
    # w, h 是内部角点 数
    w, h = 5, 7
    # w, h = 6, 8
    # w, h = 7, 9
    w, h = 3, 6
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    # store the world coord. and image coord. points. 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = []  # 在世界坐标系中的三维点
    imgpoints = []  # 在图像平面的二维点
    # images save in folder "hw6", 存放6-10张 相机拍摄的棋盘格模板的照片
    dir_1 = r'/home/zhangyong/codes/auto_calibrat/images/bascat_cams/19501458/selected'
    dir_1 = r'/home/zhangyong/codes/auto_calibrat/images/bascat_cams/test_1/1'
    images = [os.path.join(dir_1, img) for img in os.listdir(dir_1) if img[-3:] == "jpg"]
    # images = ['../images/hw6/201809261350515.jpg', '../images/hw6/201809261350513.jpg']
    # images = sorted(images)
    for fname in images:
        print(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为灰度图
        # find the corner of checkboard, 检测棋盘格的角点
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
        # save the points as long as find enough pair points
        if ret is True:
            # 亚像素角点检测,得到更为准确的角点像素坐标. (5, 5) (11, 11)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
            objpoints.append(objp)  # 3D 点坐标   (63, 3)
            imgpoints.append(corners2)   # 2D 点坐标 (63, 1, 2)
            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (w, h), corners2, ret)
            # cv2.imwrite('test/16_1.jpg', dst)
            # cv2.imshow('img',img)
            # cv2.waitKey(10)
    print(len(images), len(objpoints), len(imgpoints))
    print(imgpoints[0][10])
    # calibration, cv2.calibrateCamera这个函数会返回 标定结果、相机的内参数矩阵、畸变系数、旋转矩阵和平移向量
    imageSize = gray.shape[::-1]   # 图片尺寸,单位为像素
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imageSize, None, None)
    K = cameraMatrix  # dir_1里的图片变化了(多了,少了,换了),K就会变. 用不同的标定板,K也会变化.这些变化其实都是微调,趋向真实K值.

    # output
    print('K(内参,单位是像素) is: \n', K)
    print('distCoeffs(畸变系数) is: \n', distCoeffs)
    # print('rvecs(外参 旋转向量) is: \n', rvecs)  # 每张图一个 (3, 1)
    # print('tvecs(外参 平移向量) is: \n', tvecs)  # 每张图一个 (3, 1)
    print('相机pose num:', len(rvecs))
    rvecs_matrixs = []
    for i in range(len(rvecs)):
        rvecs_matrix, _ = cv2.Rodrigues(rvecs[i], None)  # 旋转向量和旋转矩阵 互相转换. 勒让德多项式的罗德里格斯(Rodrigues)表示式. Rodrigues变换
        # print('rvecs_matrix(外参 旋转矩阵) is: \n', i, rvecs_matrix)
        rvecs_matrixs.append(rvecs_matrix)
        info, angle = get_euler_angle(rvecs_matrix)
        t_vec = tvecs[i].reshape(1, 3)
        print('外参 旋转角度:', i, info, t_vec)

    info, angle = get_euler_angle(np.dot(rvecs_matrixs[0].T, rvecs_matrixs[1]))
    print('r_rel:', info)

    k_1 = np.array([[3.55489412e+03, 0.00000000e+00, 1.22230841e+03], [0.00000000e+00, 3.53687866e+03, 1.00094593e+03], [0,0,1]])
    d_1 = np.array([-6.75587711e-02, -7.71911226e-02, -1.42015507e-02,1.63239614e-03,  2.30015727e+00])

    ud_ref_view_pts, ud_other_view_pts = imgpoints
    ud_ref_view_pts = cv2.undistortPoints(ud_ref_view_pts, k_1, d_1, None, None)  # 去畸变
    ud_other_view_pts = cv2.undistortPoints(ud_other_view_pts, k_1, d_1, None, None)  # 去畸变
    imgpoints = [ud_ref_view_pts, ud_other_view_pts]

    # 可视化角点
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)  # 3D 点
    cam_i = 0
    # imgpts, jac = cv2.projectPoints(axis, rvecs[cam_i], tvecs[cam_i], K, distCoeffs)  # 3D->2D
    # img = cv2.imread(images[cam_i])
    # img = draw(img, imgpoints[cam_i], imgpts)   # 在标定板上画出3D坐标
    # # cv2.imshow('img', img)
    # cv2.imwrite(f'2_{cam_i}.jpg', img)
    data = [objpoints, imgpoints, images, cameraMatrix, distCoeffs, rvecs, tvecs, rvecs_matrixs]
    return data


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 2)   # 0->imgpts[0]的线
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 2)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 2)
    return img


# 从旋转向量转换为欧拉角
def get_euler_angle(R_mat):
    # calculate rotation angles  转换为角度
    rotation_vector = R.from_matrix(R_mat).as_rotvec()   # 旋转矩阵转为旋转向量
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)
    # print(f"theta:{theta}")
    # transformed to quaterniond  转换为四元数
    w = math.cos(theta / 2)
    x = math.sin(theta / 2) * rotation_vector[0] / theta
    y = math.sin(theta / 2) * rotation_vector[1] / theta
    z = math.sin(theta / 2) * rotation_vector[2] / theta

    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    # print('t0:{}, t1:{}'.format(t0, t1))
    pitch = math.atan2(t0, t1)

    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)

    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)

    # print(f'弧度: pitch:{pitch:.4f}, yaw:{yaw:.4f}, roll:{roll:.4f}')

    # 单位转换：将弧度转换为度
    # Y = int((pitch / math.pi) * 180)
    # X = int((yaw / math.pi) * 180)
    # Z = int((roll / math.pi) * 180)
    Y = (pitch / math.pi) * 180
    X = (yaw / math.pi) * 180
    Z = (roll / math.pi) * 180
    # print(f'角度: pitch:{X:.2f}, yaw:{Y:.1f}, roll:{Z:.2f}')
    info = f'角度: pitch:{X:.2f}, yaw:{Y:.1f}, roll:{Z:.2f}'
    # return 0, Y, X, Z
    return info, [X, Y, Z]


def eval_1(objpoints, imgpoints, rvecs, tvecs, K, distCoeffs):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, distCoeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("eval: total_avg_error: ", mean_error / len(objpoints))


def Error_R(r1, r2):
    '''
    Calculate isotropic rotation degree error between r1 and r2.
    :param r1: shape=(B, 3, 3), pred
    :param r2: shape=(B, 3, 3), gt
    :return:
    '''
    r2_inv = r2.transpose(0, 2, 1)
    r1r2 = np.matmul(r2_inv, r1)
    tr = r1r2[:, 0, 0] + r1r2[:, 1, 1] + r1r2[:, 2, 2]
    rads = np.arccos(np.clip((tr - 1) / 2, -1, 1))
    degrees = rads / math.pi * 180
    return degrees


def get_r_error(r1, r2):
    r1 = np.array(r1)
    r1 = r1.reshape(1, 3, 3)
    r2 = r2.reshape(1, 3, 3)
    degrees = Error_R(r1, r2)
    return degrees


def get_gt():
    print('================== camera_info gt ==================')
    json_path = '../images/bascat_cams/testall.json'
    with open(json_path) as file:
        camera_info = json.load(file)
    # print(camera_info['cameras'][16])
    cam_15 = camera_info['cameras'][15]['pmat']
    cam_16 = camera_info['cameras'][16]['pmat']

    cam_15 = np.array(cam_15).reshape(3, 4)
    cam_16 = np.array(cam_16).reshape(3, 4)
    R_15, t_15 = cam_15[0:3, 0:3], cam_15[0:3, 3]
    R_16, t_16 = cam_16[0:3, 0:3], cam_16[0:3, 3]
    # print(R_15, t_15)
    info, angle = get_euler_angle(R_15)
    print('r_15:', info, 't_15:', t_15.reshape(1, 3))
    info, angle = get_euler_angle(R_16)
    print('r_16:', info, 't_16:', t_16.reshape(1, 3))
    r_mat_rel = np.dot(R_15, R_16.T)
    info, angle = get_euler_angle(r_mat_rel)  # cam_16--> cam_15
    print('r_rel:', info, 't_rel:', (t_15 - t_16).reshape(1, 3), 't_rel_norm', normalize((t_15 - t_16).reshape(1, 3)))
    print("r_mat_rel:", r_mat_rel)
    return camera_info
    