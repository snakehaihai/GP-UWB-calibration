from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import scipy.optimize
from itertools import product
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import least_squares
import multiprocessing

import sys
from contextlib import redirect_stdout

save_dir = "/home/lbd/slam/Lio/ntu/makern35/2/down_sample2/0926_2239_log10/7_3/"

def process_item(i):
    save_log_path = save_dir + f"log_{i}.txt"
    with open(save_log_path, 'w') as f:
        with redirect_stdout(f):
            print("num is : ", i , " , name is : ", cdis_0_all_name[i])
            dis1_col = []
            dis0_col = []
            for loader in loader_list:
                dis0_col.append(loader.get_uwb_col(cdis_0_all_name[i]))
                dis1_col.append(loader.get_uwb_col(cdis_1_all_name[i]))

            now0_col = np.concatenate(dis0_col, axis=0)
            print(i, " now0_col :", len(now0_col!=0))
            if(i==1):
                voxel_size = 0.01
                r = 100
                rmax = 100
                rmin = 11
                filtered_X0, filtered_dis0 = voxel_downsampling(gtp0, now0_col, voxel_size, r,save_dir + f"comparison_scatter_gp0_{i}_{voxel_size}_{r}.png")
                filtered_X0, filtered_dis0 = stratified_sampling(filtered_X0, filtered_dis0, 50, 50 , rmin, rmax, f"comparison_str_scatter_gp0_{i}_{rmin}_{rmax}.png")
                # filtered_X0, filtered_dis0 = voxel_downsampling(gtp0, now0_col, voxel_size, r,save_dir + f"comparison_scatter_gp0_{i}_{voxel_size}_{r}.png")
            elif(i==3):
                voxel_size = 0.01
                r = 60
                rmax = 40
                rmin = 10
                filtered_X0, filtered_dis0 = voxel_downsampling(gtp0, now0_col, voxel_size, r,save_dir + f"comparison_scatter_gp0_{i}_{voxel_size}_{r}.png")
                filtered_X0, filtered_dis0 = stratified_sampling(filtered_X0, filtered_dis0, 50, 50 , rmin, rmax, f"comparison_str_scatter_gp0_{i}_{rmin}_{rmax}.png")
                # filtered_X0, filtered_dis0 = voxel_downsampling(gtp0, now0_col, voxel_size, r,save_dir + f"comparison_scatter_gp0_{i}.png")
            elif(i==6):
                voxel_size = 0.05
                r = 25
                r = 100
                rmax = 100
                rmin = 9
                filtered_X0, filtered_dis0 = voxel_downsampling(gtp0, now0_col, voxel_size, r,save_dir + f"comparison_scatter_gp0_{i}_{voxel_size}_{r}.png")
                filtered_X0, filtered_dis0 = stratified_sampling(filtered_X0, filtered_dis0, 50, 50 , rmin, rmax, f"comparison_str_scatter_gp0_{i}_{rmin}_{rmax}.png")
                # filtered_X0, filtered_dis0 = voxel_downsampling(gtp0, now0_col, 0.02, 120,save_dir + f"comparison_scatter_gp0_{i}.png")
            elif(i==7):
                voxel_size = 0.05
                r = 25
                r = 100
                rmax = 100
                rmin = 6
                filtered_X0, filtered_dis0 = voxel_downsampling(gtp0, now0_col, voxel_size, r,save_dir + f"comparison_scatter_gp0_{i}_{voxel_size}_{r}.png")
                filtered_X0, filtered_dis0 = stratified_sampling(filtered_X0, filtered_dis0, 50, 50 , rmin, rmax, f"comparison_str_scatter_gp0_{i}_{rmin}_{rmax}.png")
                # filtered_X0, filtered_dis0 = voxel_downsampling(gtp0, now0_col, 0.02, 120,save_dir + f"comparison_scatter_gp0_{i}.png")
            else:
                voxel_size = 0.05
                r = 25
                r = 100
                rmax = 100
                rmin = 6
                filtered_X0, filtered_dis0 = voxel_downsampling(gtp0, now0_col, voxel_size, r,save_dir + f"comparison_scatter_gp0_{i}_{voxel_size}_{r}.png")
                filtered_X0, filtered_dis0 = stratified_sampling(filtered_X0, filtered_dis0, 50, 50 , rmin, rmax, f"str_scatter_gp0_{i}_{rmin}_{rmax}.png")
                # filtered_X0, filtered_dis0 = voxel_downsampling(gtp0, now0_col, 0.02, 40,save_dir + f"comparison_scatter_gp0_{i}.png")
            r = 120 # MCD1
            # r = 200 #MCD2
            # filtered_X0, filtered_dis0 = voxel_downsampling(gtp0, now0_col, voxel_size, r,save_dir + f"comparison_scatter_gp0_{i}_{voxel_size}_{r}.png")
            # filtered_X0, filtered_dis0 = stratified_sampling(filtered_X0, filtered_dis0, 50, 50 , 0.2, r)
                # filtered_X0, filtered_dis0 = stratified_sampling(gtp0, now0_col, 15, 100 , 1, 120)#1
            # filtered_X0, filtered_dis0 = stratified_sampling(gtp0, now0_col, 50, 25 , 0.2, 40)#1
            min_0 = np.min(filtered_dis0)
            min_index = np.unravel_index(np.argmin(filtered_dis0), filtered_dis0.shape)
            min_point = filtered_X0[min_index]
            print(i,"rmin max r: ",rmin, rmax, r)
            print(i,"min_0 : ", min_0)
            print(i,"min_index : ", min_index)
            print(i,"min gt0: ", min_point)
            
            print(i,"orin   max: ", np.max(filtered_dis0), ", min :", np.min(filtered_dis0), ", mean :", np.mean(filtered_dis0))
            # filtered_dis0 = 1 / np.log10(filtered_dis0+1) # log10
            # filtered_dis0 = 1 / np.log(filtered_dis0 + 1)
            # filtered_dis0 = 1 / np.power(filtered_dis0, 1/4)
            # filtered_dis0 = 1 / filtered_dis0 # 1 3 6
            # filtered_dis0 = 1 / np.power(filtered_dis0, 1/3)
            # filtered_dis0 = 1 / np.sqrt(filtered_dis0 + 1) # without 1 3 6
            filtered_dis0 =  1/np.log10(np.sqrt(filtered_dis0) + 1)
            # print(i,"filter max: ", np.max(filtered_dis0), ", min :", np.min(filtered_dis0), ", mean :", np.mean(filtered_dis0))
            print(i,"filtered_X0.shape", filtered_X0.shape)

            now1_col = np.concatenate(dis1_col, axis=0)
            if(i==1):
                voxel_size = 0.01
                r = 120
                rmin = 11
                rmax = 120
                filtered_X1, filtered_dis1 = voxel_downsampling(gtp1, now1_col, voxel_size, r,save_dir + f"comparison_scatter_gp1_{i}_{voxel_size}_{r}.png")
                filtered_X1, filtered_dis1 = stratified_sampling(filtered_X1, filtered_dis1, 50, 60 , rmin, rmax, f"str_scatter_gp1_{i}_{rmin}_{rmax}.png")#1
                # filtered_X0, filtered_dis0 = voxel_downsampling(gtp0, now0_col, voxel_size, r,save_dir + f"comparison_scatter_gp0_{i}_{voxel_size}_{r}.png")
            elif(i==3):
                voxel_size = 0.01
                r = 40
                rmin = 10
                rmax = 40
                filtered_X1, filtered_dis1 = voxel_downsampling(gtp1, now1_col, voxel_size, r,save_dir + f"comparison_scatter_gp1_{i}_{voxel_size}_{r}.png")
                filtered_X1, filtered_dis1 = stratified_sampling(filtered_X1, filtered_dis1, 50, 60 , rmin, rmax, f"str_scatter_gp1_{i}_{rmin}_{rmax}.png")#1
                # filtered_X0, filtered_dis0 = voxel_downsampling(gtp0, now0_col, voxel_size, r,save_dir + f"comparison_scatter_gp0_{i}.png")
            elif(i==7):
                voxel_size = 0.05
                r = 120
                rmin = 9
                rmax = 120
                filtered_X1, filtered_dis1 = voxel_downsampling(gtp1, now1_col, voxel_size, r,save_dir + f"comparison_scatter_gp1_{i}_{voxel_size}_{r}.png")
                filtered_X1, filtered_dis1 = stratified_sampling(filtered_X1, filtered_dis1, 50, 60 , rmin, rmax, f"str_scatter_gp1_{i}_{rmin}_{rmax}.png")#1
                # filtered_X0, filtered_dis0 = voxel_downsampling(gtp0, now0_col, 0.02, 120,save_dir + f"comparison_scatter_gp0_{i}.png")
            else:
                voxel_size = 0.05
                r = 120
                rmin = 9
                rmax = 100
                filtered_X1, filtered_dis1 = voxel_downsampling(gtp1, now1_col, voxel_size, r,save_dir + f"comparison_scatter_gp1_{i}_{voxel_size}_{r}.png")
                filtered_X1, filtered_dis1 = stratified_sampling(filtered_X1, filtered_dis1, 50, 60 , rmin, rmax, f"str_scatter_gp1_{i}_{rmin}_{rmax}.png")#1
            r = 100 #MCD1
            # r = 200 #MCD2
            voxel_size = 0.02

            # filtered_X1, filtered_dis1 = stratified_sampling(gtp1, now1_col, 35, 15 , 20, 80)#1
            # filtered_X1, filtered_dis1 = process_data_data_dis(gtp1, now1_col, 30, 200)
            # filtered_X1, filtered_dis1 = stratified_sampling(gtp1, now1_col, 50, 30 , 20, 120)#1
            min_1 = np.min(filtered_dis1)
            min_index = np.unravel_index(np.argmin(filtered_dis1), filtered_dis1.shape)
            min_point1 = filtered_X1[min_index]

            print(i,"min_1 : ", min_1)
            print(i,"min_index : ", min_index)
            print(i,"min gt1: ", min_point1)
            # filtered_dis1 = 1 / np.log10(filtered_dis1+1)
            filtered_dis1 =  1 / np.log10(np.sqrt(filtered_dis1) + 1)
            # filtered_dis1 = 1 / np.log(filtered_dis1+1)
            # filtered_dis1 = 1 / np.power(filtered_dis1, 1/4)
            # filtered_dis1 = 1 / filtered_dis1 # 1 3 6
            # filtered_dis1 = 1 / filtered_dis1 # 1 3 6
            # filtered_dis1 = 1 / np.power(filtered_dis1, 1/3)
            # filtered_dis1 = 1 / np.sqrt(filtered_dis1 + 1)
            print(i,"filtered_X1.shape", filtered_X1.shape)

            concatenated_X = np.concatenate((filtered_X0, filtered_X1), axis=0)
            concatenated_dis = np.concatenate((filtered_dis0, filtered_dis1), axis=0)

            print(i,"concatenated_X.shape", concatenated_X.shape)
            print(i,"concatenated_dis.shape", concatenated_dis.shape)
            mean_dis = np.mean(concatenated_dis)
            print(f"Mean of dis: {mean_dis}")
            #针对下采样后的点进行桶采集
            # concatenated_X, concatenated_dis = stratified_sampling(concatenated_X, concatenated_dis, 80, 50 , 0.1, r)#1
            print(i,"voxel_down_concatenated_X.shape", concatenated_X.shape)
            print(i,"voxel_down_concatenated_dis.shape", concatenated_dis.shape)
            sorted_indices = np.argsort(concatenated_dis)
            sorted_dis = np.array(concatenated_dis)[sorted_indices]
            sorted_pos = np.array(concatenated_X)[sorted_indices]
            index = next(i for i, d in enumerate(sorted_dis) if d > mean_dis)

            corresponding_pos = sorted_pos[index]
            print(f"Index of first value greater than mean in sorted dis: {index}")
            print(f"Corresponding pos: {corresponding_pos}")

            if concatenated_X.shape[0] < 10:
                print("data num is too low!")
                return  # or handle as needed

            np.save(save_dir + f'concatenated_X_{i}.npy', concatenated_X)
            np.save(save_dir + f'concatenated_dis_{i}.npy', concatenated_dis)
            # kernel = Matern(length_scale=1, length_scale_bounds=(1e-3, 1e3), nu=5/3) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-20, 1e2))
            kernel = Matern(length_scale=0.5, length_scale_bounds=(1e-4, 1e3), nu=5/3) + WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-20, 1e3))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, optimizer="fmin_l_bfgs_b")# max_iter_predict=1000)

            print("time to fit anchor :", i)
            time_now = time.time()

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            pipeline = Pipeline(steps=[
                ('preprocessor', numeric_transformer),
                ('regressor', gp)
            ])

            X_train = concatenated_X
            y_train = concatenated_dis
            print(i,"X_train shape", X_train.shape)
            print(i,"y_train shape", y_train.shape)

            
            pipeline.fit(X_train, y_train)
            time_last = time.time()
            print(i," time cost :", time_last-time_now)
            # model_filename = '/home/lbd/slam/Lio/ntu/makern35/2/figure/3/new/pipeline_model3D_sqrtinv31_2_' + str(i) + '.pkl'
            # model_filename = '/home/lbd/slam/Lio/ntu/makern35/2/down_sample/pipeline_model3D_sqrtinv31_2_' + str(i) + '.pkl' #MCD2
            model_filename = save_dir + '/pipeline_model3D_sqrtinv_' + str(i) + '.pkl' #MCD1
            # model_filename = '/home/lbd/slam/Lio/ntu/makern35/2/figure/3/3/sqrt_random_sample2/fig_again/fig_again/pipeline_model3D_sqrtinv' + str(i) + '.pkl'
            # model_filename = '/home/lbd/slam/Lio/ntu/makern35/2/figure/3/3/sqrt_random_sample/pipeline_model3D_sqrtinv' + str(i) + '.pkl'
            joblib.dump(pipeline, model_filename)
            # 这里你可以选择保存模型：pipeline.fit(X_train, y_train), 或者是训练模型并进行其它操作


#
class uwb_loader():
    def __init__(self, uwb_path):
        self.uwb_data = pd.read_csv(uwb_path)
        self.uwb_path = uwb_path
        
    def get_true_coordintates(self):
        coordinates0 = self.uwb_data[['pt0x', 'pt0y', 'pt0z']].to_numpy()
        coordinates1 = self.uwb_data[['pt1x', 'pt1y', 'pt1z']].to_numpy()
        return coordinates0, coordinates1
    
    def read_uwb(self, uwb_path):
        self.uwb_data = pd.read_csv(uwb_path)
        self.uwb_path = uwb_path
        return self.uwb_data
    
    def get_uwb_data(self):
        return self.uwb_data
    
    
    def get_uwb_col(self, name):
        if(name in self.uwb_data.columns):
            return self.uwb_data[name]
        else:
            print("name: ", name, "# is not in the colums")
            print("colums: ", self.uwb_data.colums)
            return 
    
    def get_uwb_col_concate(self, name):
        return self.uwb_data[name].to_numpy()
        
    
class slam_loader():
    def __init__(self,slam_path, tum_or_log, R , t) -> None:
        self.slam_path = slam_path
        # global frame
        self.R = R
        self.t = t
        if(tum_or_log):
            self.slam_data = self.read_slam_with_tum(slam_path, R , t)
        else:
            self.slam_data = self.read_slam_with_ntulog(slam_path, R , t)
    
    def read_slam_with_tum(self, slam_tum_path, R , t):
        column_names = ['time', 'x', 'y','z','qw','qx','qy','qz']
        slam_data = pd.read_csv(slam_tum_path, delimiter=' ', names=column_names)
        return slam_data

    def read_slam_with_ntulog(self, slam_ntu_logpath, R ,t):
        slam_data = pd.read_csv(slam_ntu_logpath)
        return slam_data
    #tum
    def get_slam_coordintates(self):
        coordinates = self.slam_data[['x', 'y', 'z']].to_numpy()
        return coordinates

    def get_slam_data(self):
        return self.slam_data

    def get_slam_col(self, name):
        if(name in self.slam_data.columns):
            return self.slam_data[name]
        else:
            print("name: ", name, "# is not in the colums")
            print("colums: ", self.slam_data.colums)
            return 

    def get_slam_col_concate(self, name):
        return self.slam_data[name].to_numpy()
#uwb 对齐到slam，以slam shape为目标
def process_slam_timestamp(uwb_data, slam_data):
    # todo
    # 数据结构应该是
    # uwb time                            dis
    #      1644823131.49023
    # slam time                           x y z
    #      1644823131.888480
    # 对位姿进行插值,与uwb时间戳对齐
    t1 = 0
    t2 = 1
    i = 0
    slam_uwb_data = []
    slam_data_change = []
    while(i < len(slam_data)):
        try:
            time0 = uwb_data[t1][0]
            # time1 = uwb_data[t2][0]
            # print("time uwb is : ", time0)
        except:
            print("get data failed")
            return np.array(slam_uwb_data), np.array(slam_data_change)
        
        if(slam_data[i][0] < time0 - 0.03):
            i = i+1
        
        elif(slam_data[i][0] > time0 + 0.03):
            t1 = t1+1   

        else:
            slam_uwb_data.append(uwb_data[t1])
            slam_data_change.append(slam_data[i])
            i = i + 1
            t1 = t1 + 1
    
    return p.array(slam_uwb_data), np.array(slam_data)
#######################################         RT         ###################
# <group if="$(eval 'ntu_day_01' in bag_file)">  <rosparam param="/tf_Lprior_L0_init"> [ 49.28,  107.38, 7.58, -41,  0,  0] </rosparam> </group>
# <group if="$(eval 'ntu_day_02' in bag_file)">  <rosparam param="/tf_Lprior_L0_init"> [ 61.99,  119.58, 7.69, -134, 0,  0] </rosparam> </group>
# <group if="$(eval 'ntu_day_03' in bag_file)">  <rosparam param="/tf_Lprior_L0_init"> [ 62.82,  119.51, 7.70, 39,   0, -0] </rosparam> </group>
# <group if="$(eval 'ntu_day_04' in bag_file)">  <rosparam param="/tf_Lprior_L0_init"> [ 55.52,  110.70, 7.72, -40,  0,  0] </rosparam> </group>
# <group if="$(eval 'ntu_day_05' in bag_file)">  <rosparam param="/tf_Lprior_L0_init"> [ 59.18,  116.06, 7.72, 42,   0,  0] </rosparam> </group>
# <group if="$(eval 'ntu_day_06' in bag_file)">  <rosparam param="/tf_Lprior_L0_init"> [ 48.67,  109.16, 7.64, -28,  0,  0] </rosparam> </group>
# <group if="$(eval 'ntu_day_07' in bag_file)">  <rosparam param="/tf_Lprior_L0_init"> [-27.11,  -1.57,  8.73, -8,   0,  0] </rosparam> </group>
# <group if="$(eval 'ntu_day_08' in bag_file)">  <rosparam param="/tf_Lprior_L0_init"> [ 40.58,  15.90,  6.56, 48,   0,  0] </rosparam> </group>
# <group if="$(eval 'ntu_day_09' in bag_file)">  <rosparam param="/tf_Lprior_L0_init"> [ 71.90,  57.99,  7.67, 80,   0,  0] </rosparam> </group>
# <group if="$(eval 'ntu_night_01' in bag_file)">  <rosparam param="/tf_Lprior_L0_init"> [59.69,  108.43,  7.82, -36,  0, 0] </rosparam> </group>
# <group if="$(eval 'ntu_night_02' in bag_file)">  <rosparam param="/tf_Lprior_L0_init"> [55.78,  108.37,  7.78, -32,  0, 0] </rosparam> </group>
# <group if="$(eval 'ntu_night_03' in bag_file)">  <rosparam param="/tf_Lprior_L0_init"> [143.28, 36.80,   8.97, -136, 0, 0] </rosparam> </group>
# <group if="$(eval 'ntu_night_04' in bag_file)">  <rosparam param="/tf_Lprior_L0_init"> [244.20, -99.86,  5.97, -32,  0, 1] </rosparam> </group>
# <group if="$(eval 'ntu_night_05' in bag_file)">  <rosparam param="/tf_Lprior_L0_init"> [85.37,  73.99,   7.77, -132, 0, 0] </rosparam> </group>
# <group if="$(eval 'ntu_night_06' in bag_file)">  <rosparam param="/tf_Lprior_L0_init"> [46.02,  021.03,  6.6,  -135, 0, 0] </rosparam> </group>
# <group if="$(eval 'ntu_night_07' in bag_file)">  <rosparam param="/tf_Lprior_L0_init"> [55.97,  112.70,  7.75, -36,  0, 0] </rosparam> </group>
# <group if="$(eval 'ntu_night_09' in bag_file)">  <rosparam param="/tf_Lprior_L0_init"> [234.26, -41.31,  6.69, -107, 0, 0] </rosparam> </group>
# <group if="$(eval 'ntu_night_10' in bag_file)">  <rosparam param="/tf_Lprior_L0_init"> [194.55, -216.91, -3.69, 176, 0, 0] </rosparam> </group>
# <group if="$(eval 'ntu_night_11' in bag_file)">  <rosparam param="/tf_Lprior_L0_init"> [15.34,  -197.79, -4.99, 124, 0, 0] </rosparam> </group>
# <group if="$(eval 'ntu_night_12' in bag_file)">  <rosparam param="/tf_Lprior_L0_init"> [60.77,  -45.23,   2.2, -139, 0, 0] </rosparam> </group>
# <group if="$(eval 'ntu_night_13' in bag_file)">  <rosparam param="/tf_Lprior_L0_init"> [81.38,  -18.45,   3.43, 42,  0, 0] </rosparam> </group>
#process 3D coordintates time to fit the uwb measuanments time ntu_night_04

# 欧拉角（例如，用弧度表示）
euler_angles_degrees = [42, 0, 0]
euler_angles_radians = np.deg2rad(euler_angles_degrees)
r = R.from_euler('zyx', euler_angles_radians)
rotation_matrix = r.as_matrix()
print("Rotation matrix:")
print(rotation_matrix)
# 平移
t = np.array([49.28,  107.38, 7.58])
# to_body   
# ltpb_tag0:
#     T:
#     - [1.0, 0.0, 0.0, -0.0412669]
#     - [0.0, 1.0, 0.0, 0.412464]
#     - [0.0, 0.0, 1.0, 0.0951186]
#     - [0.0, 0.0, 0.0, 1.0]
# ltpb_tag1:
#     T:
#     - [1.0, 0.0, 0.0, -0.0576088]
#     - [0.0, 1.0, 0.0, -0.38734]
#     - [0.0, 0.0, 1.0, 0.117334]
#     - [0.0, 0.0, 0.0, 1.0]

############################################################################train begin#######################################################
train_data_list = [
"MCDUWB/train/ntu_day_01_packed.csv",
"MCDUWB/train/ntu_day_03_packed.csv",
"MCDUWB/train/ntu_day_04_packed.csv",
"MCDUWB/train/ntu_day_05_packed.csv",
"MCDUWB/train/ntu_day_06_packed.csv",
"MCDUWB/train/ntu_day_07_packed.csv",
"MCDUWB/train/ntu_day_08_packed.csv",
"MCDUWB/train/ntu_day_09_packed.csv",
"MCDUWB/train/ntu_night_01_packed.csv",
"MCDUWB/train/ntu_night_03_packed.csv",
"MCDUWB/train/ntu_night_05_packed.csv",
"MCDUWB/train/ntu_night_06_packed.csv" ,
"MCDUWB/train/ntu_night_07_packed.csv" ,
"MCDUWB/train/ntu_night_08_packed.csv" ,
"MCDUWB/train/ntu_night_09_packed.csv" ,
"MCDUWB/train/ntu_night_10_packed.csv",
"MCDUWB/train/ntu_night_11_packed.csv",
"MCDUWB/train/ntu_night_12_packed.csv"]

# train_data_list2 = [
# "MCDUWB2/train/ntu_day_01_packed.csv",
# "MCDUWB2/train/ntu_day_03_packed.csv",
# "MCDUWB2/train/ntu_day_04_packed.csv",
# "MCDUWB2/train/ntu_day_05_packed.csv",
# "MCDUWB2/train/ntu_day_06_packed.csv",
# "MCDUWB2/train/ntu_day_07_packed.csv",
# "MCDUWB2/train/ntu_day_08_packed.csv",
# "MCDUWB2/train/ntu_day_09_packed.csv",
# "MCDUWB2/train/ntu_night_01_packed.csv",
# "MCDUWB2/train/ntu_night_03_packed.csv",
# "MCDUWB2/train/ntu_night_05_packed.csv",
# "MCDUWB2/train/ntu_night_06_packed.csv" ,
# "MCDUWB2/train/ntu_night_07_packed.csv" ,
# "MCDUWB2/train/ntu_night_08_packed.csv" ,
# "MCDUWB2/train/ntu_night_09_packed.csv" ,
# "MCDUWB2/train/ntu_night_10_packed.csv",
# "MCDUWB2/train/ntu_night_11_packed.csv",
# "MCDUWB2/train/ntu_night_12_packed.csv"]

uwb_path_ntu_day_01 = "./MCDUWB/train/ntu_day_01_packed.csv"
uwb_path_ntu_day_03 = "./MCDUWB/train/ntu_day_03_packed.csv"
uwb_path_ntu_day_04 = "./MCDUWB/train/ntu_day_04_packed.csv"
slam_FLO_STD_base = "./MCDUWB/slam/FLO-STD/"
slam_seq = "ntu_01_day/en_loop/result.csv"
#01[ 49.28,  107.38, 7.58, -41,  0,  0]
slam_path = slam_FLO_STD_base + slam_seq
muwb_loader_ntu_day_01 = uwb_loader(uwb_path_ntu_day_01)
muwb_loader_ntu_day_03 = uwb_loader(uwb_path_ntu_day_03)
muwb_loader_ntu_day_04 = uwb_loader(uwb_path_ntu_day_04)
loader_list = []
current_path = os.path.dirname(os.path.abspath(__file__))
for path_data in train_data_list:
# for path_data in train_data_list2:
    muwb_loader_ntu = uwb_loader(current_path + "/" + path_data)
    loader_list.append(muwb_loader_ntu)
    
print("read ", len(loader_list), " data! ")
loader_len = len(loader_list)
mslam_loader = slam_loader(slam_path, True, rotation_matrix, t)
# read data
uwb_data_day_01 = muwb_loader_ntu_day_01.get_uwb_data()
uwb_data_day_03 = muwb_loader_ntu_day_03.get_uwb_data()
uwb_data_day_04 = muwb_loader_ntu_day_04.get_uwb_data()
# print("uwb_data:\n",uwb_data)
slam_data = mslam_loader.get_slam_data()
# print("slam_data:\n",slam_data)

# gt_pos
day1gtp0, day1gtp1 = muwb_loader_ntu_day_01.get_true_coordintates()
day3gtp0, day3gtp1 = muwb_loader_ntu_day_03.get_true_coordintates()
day4gtp0, day4gtp1 = muwb_loader_ntu_day_04.get_true_coordintates()
print("day1gtp0.shape:",day1gtp0.shape)
print("day3gtp0.shape:",day3gtp0.shape)
print("day4gtp0.shape:",day4gtp0.shape)
datagtp0 = []
datagtp1 = []
for i in range(loader_len):
    loader = loader_list[i]
    gtp0_ , gtp1_ = loader.get_true_coordintates()
    datagtp0.append(gtp0_)
    datagtp1.append(gtp1_)
# gtp0 = np.concatenate([day1gtp0 , day3gtp0, day4gtp0 ],axis=0)
# gtp1 = np.concatenate([day1gtp1 , day3gtp1, day4gtp1 ],axis=0)
gtp0 = np.concatenate(datagtp0 ,axis=0)
gtp1 = np.concatenate(datagtp1 ,axis=0)
# print("gtp0:\n",gtp0)
print("gtp0.shape:",gtp0.shape)

# slam pos
# turn the pos to global frame using 
slam_pos = mslam_loader.get_slam_coordintates()
slam_Rt = (rotation_matrix @ slam_pos.transpose() + t.reshape(3,1)).transpose()
slam_time = mslam_loader.get_slam_col("time").to_numpy()

# 使用 numpy.insert 方法插入 slam_time 到 slam_Rt 的前面
time_RT_array = np.insert(slam_Rt, 0, slam_time, axis=1)
print(time_RT_array[0])
# print("slam_Rt:\n",slam_Rt)
# print("slam_Rt_shape:\n",slam_Rt.shape)
# print("slam_t_rt shape:",time_RT_array.shape)

# print("slam_pos:\n",slam_pos)
# print("slam_pos_shape:\n",slam_pos.shape)
cdis_0_0 = muwb_loader_ntu_day_01.get_uwb_col("dis_0_0")
cdis_time = muwb_loader_ntu_day_01.get_uwb_col("# t").to_numpy()
# print("cdis_0_0:\n",cdis_0_0.to_numpy())
# print("cdis_0_0_shape:\n",cdis_0_0.to_numpy().shape)

# rss and dis for tag 0 1
crss_0_all_name = ["rss_0_0","rss_0_1","rss_0_2","rss_0_3","rss_0_4","rss_0_5","rss_0_6","rss_0_7","rss_0_8","rss_0_9"]
cdis_0_all_name = ["dis_0_0","dis_0_1","dis_0_2","dis_0_3","dis_0_4","dis_0_5","dis_0_6","dis_0_7","dis_0_8","dis_0_9"]
cdis_1_all_name = ["dis_1_0","dis_1_1","dis_1_2","dis_1_3","dis_1_4","dis_1_5","dis_1_6","dis_1_7","dis_1_8","dis_1_9"]
cdis_0_all = muwb_loader_ntu_day_01.get_uwb_col_concate(cdis_0_all_name)
time_dis0_array = np.insert(cdis_0_all, 0, cdis_time, axis=1)
# print("cdis_0_all:\n",cdis_0_all)
# print("cdis_0_all.shape:",cdis_0_all.shape)
#取最近临时刻
data_uwb_train, data_slam_train = process_slam_timestamp(time_dis0_array, time_RT_array)
print("data_uwb_train shape is: ", data_uwb_train.shape)
print("data_slam_train shape is: ", data_slam_train.shape)
new_data_uwb_train= data_uwb_train[:, 1:]
new_data_slam_train= data_slam_train[:, 1:]
print("new_data_uwb_train shape is: ", new_data_uwb_train.shape)
print("new_data_slam_train shape is: ", new_data_slam_train.shape)
# get the min dis
min_values = []
min_indices = []
pos_x = []
for i, row in enumerate(cdis_0_all):
    # 筛选出非0元素
    non_zero_elements = row[row != 0]
    if len(non_zero_elements) > 0:
        # 如果存在非0元素，找出非0元素中的最小值及其索引
        min_value = np.min(non_zero_elements)
        min_index = np.where(row == min_value)[0][0]
        # 将结果添加到结果列表中
        min_values.append(min_value)
        min_indices.append(min_index)
        pos_x.append(gtp0[i])
    else:
        # 如果全是0，设定最小值为0，索引不存在非0的情况下为 -1 或其他值（依据具体需求）
        min_value = 0
        min_index = -1  # 此处用 -1 表示没有非0元素存在，具体业务需求可调整

print("pos_x : ", len(pos_x))
print("min_values : ", len(min_values))


# get the min uwb_slam
min_values2 = []
min_indices2 = []
pos_x2 = []
for i, row in enumerate(new_data_uwb_train):
    # 筛选出非0元素
    non_zero_elements = row[row != 0]
    if len(non_zero_elements) > 0:
        # 如果存在非0元素，找出非0元素中的最小值及其索引
        min_value = np.min(non_zero_elements)
        min_index = np.where(row == min_value)[0][0]
        # 将结果添加到结果列表中
        min_values2.append(min_value)
        min_indices2.append(min_index)
        pos_x2.append(new_data_slam_train[i])
    else:
        # 如果全是0，设定最小值为0，索引不存在非0的情况下为 -1 或其他值（依据具体需求）
        min_value = 0
        min_index = -1  # 此处用 -1 表示没有非0元素存在，具体业务需求可调整


print("pos_x2 : ", len(pos_x2))
print("min_values2 : ", len(min_values2))
#evey 10anchor
# 定义损失函数
def distance_difference(beacon_position, pos, distances, gp):
    beacon_position = np.array(beacon_position).reshape(1, -1)
    predicted_distances, _ = gp.predict(pos, return_std=True)
    loss = np.mean((predicted_distances - distances)**2)
    return loss

all_concatenated_X=[]
all_concatenated_dis=[]
# 初始化基站的位置（可以随机初始化或通过其他方法初始化）
num_beacons = 10
beacon_positions=[]
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
dis_thr = 30
def process_data_data_dis(pos_data_ ,dis_, dis_begin = 0 ,dis_thr = 50):
    #大于dis_begin < dis_thr
    indices = np.where((dis_ > dis_begin) & (dis_ < dis_thr))[0]
    pos_data = pos_data_[indices]
    dis = dis_[indices]
    return pos_data, dis

def plot_2d_scatter(pos_data, sampled_pos_data, title, file_name):
    """
    绘制2D散点图，包含原始数据和下采样数据，使用不同颜色区分。
    
    参数:
    pos_data (np.ndarray): 原始位置信息数组，形状为 (N, 3)。
    sampled_pos_data (np.ndarray): 下采样后的位置信息数组，形状为 (M, 3)。
    title (str): 图表标题。
    file_name (str): 保存图像的文件名。
    """
    plt.figure(figsize=(10, 8))
    
    # 绘制原始数据，使用蓝色
    plt.scatter(pos_data[:, 0], pos_data[:, 1], s=2, color='blue', label='ori_data')
    
    # 绘制下采样数据，使用红色
    plt.scatter(sampled_pos_data[:, 0], sampled_pos_data[:, 1], s=1, color='red', label='over_data')

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()  # 添加图例
    plt.savefig(file_name)
    plt.close()  # 关闭图像以释放内存

def voxel_downsampling(pos_data_, distances_, voxel_size=0.02, max_dis=100, file_name="abc"):
    """
    对数据进行体素下采样，每个体素内只保留一个位置信息及其对应的距离。
    
    参数:
    pos_data_ (np.ndarray): 包含位置信息的数组，形状为 (N, 3)。
    distances_ (np.ndarray): 与位置信息对应的距离数组，形状为 (N,)。
    voxel_size (float): 体素网格的大小，默认为0.05。
    max_dis (float): 最大距离过滤阈值，默认为75。
    file_name (str): 保存图像的文件名。
    
    返回:
    np.ndarray: 体素下采样后的位置信息数组。
    np.ndarray: 体素下采样后的距离数组。
    """
    print("原始数据形状: ", pos_data_.shape)
    
    # 过滤掉 r <= 0 和 r >= max_dis 的数据
    valid_indices = np.where((distances_ > 0) & (distances_ < max_dis))[0]
    pos_data = pos_data_[valid_indices]
    distances = distances_[valid_indices]
    print("过滤后数据形状: ", pos_data.shape)
    print("过滤后最大最小值: ", np.max(distances), " ",np.min(distances))
    # # 将位置信息转换为体素网格坐标
    voxel_indices = np.floor(pos_data / voxel_size).astype(int)

    # 使用字典存储每个体素的第一个点和对应的距离
    voxel_dict = {}
    for i, voxel_index in enumerate(voxel_indices):
        voxel_key = tuple(voxel_index)
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key] = (pos_data[i], distances[i])

    # 提取字典中的位置信息和距离信息
    sampled_pos_data = np.array([v[0] for v in voxel_dict.values()])
    sampled_distances = np.array([v[1] for v in voxel_dict.values()])
    # sampled_pos_data = pos_data
    # sampled_distances = distances
    print("下采样后数据形状: ", sampled_pos_data.shape)

    # 绘制2D散点图
    plot_2d_scatter(pos_data, sampled_pos_data, "compare_data", file_name)

    return sampled_pos_data, sampled_distances


def stratified_sampling(pos_data_, distances_, num_bins=10, samples_per_bin=100, min_dis = 0, max_dis = 75, file_name = ""):
    """
    对数据进行分层采样，使数据在各个距离区间内更加均匀分布。
    
    参数:
    pos_data (np.ndarray): 包含位置信息的数组。
    distances (np.ndarray): 与位置信息对应的距离数组。
    num_bins (int): 距离区间的数量，默认值为10。
    samples_per_bin (int): 每个距离区间内的样本数量，默认值为100。
    
    返回:
    np.ndarray: 分层采样后的位置信息数组。
    np.ndarray: 分层采样后的距离数组。
    """
    # 获取距离的最小值和最大值
    indices = np.where((distances_ > min_dis) & (distances_ < max_dis))[0]
    pos_data = pos_data_[indices]
    distances = distances_[indices]
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    # min_distance = min_dis
    # max_distance = max_dis
    # 计算每个距离区间的边界
    bins = np.linspace(min_distance, max_distance, num_bins + 1)
    
    # 存储采样后的数据
    sampled_pos_data = []
    sampled_distances = []
    
    # 在每个距离区间内进行采样
    for i in range(num_bins):
        bin_indices = np.where((distances >= bins[i]) & (distances < bins[i + 1]))[0]
        if len(bin_indices) > 0:
            # 如果区间内的数据量大于所需样本量，则进行随机采样
            if len(bin_indices) > samples_per_bin:
                sampled_indices = np.random.choice(bin_indices, samples_per_bin, replace=False)
            else:
                sampled_indices = bin_indices
            sampled_pos_data.append(pos_data[sampled_indices])
            sampled_distances.append(distances[sampled_indices])
            # print(" i: ",i , "  samples num: ", len(sampled_indices))
    
    
    # 将采样后的数据拼接起来
    sampled_pos_data = np.concatenate(sampled_pos_data)
    sampled_distances = np.concatenate(sampled_distances)
    # 绘制2D散点图
    plot_2d_scatter(pos_data, sampled_pos_data, "compare_data2", file_name)
    print("sampled : ", len(sampled_pos_data))
    return sampled_pos_data, sampled_distances
 
######################################################################################################one gp-> one anchor
max_arg_max = []
x_ = 400
y_ = 400
z_ = 100
#范围
# x_r_max = 400
# x_r_min = -200
# y_r_max = 200
# y_r_min = -400
#坐标轴调转
x_r_max = -200
x_r_min = 400
y_r_max = -400
y_r_min = 200
z_r_min = -100
z_r_max = 100
x10 = np.linspace(x_r_min, x_r_max, x_)
x20 = np.linspace(y_r_min, y_r_max, y_)
x30 = np.linspace(z_r_min, z_r_max, z_)
x1x20x30 = np.array([0]*(y_*x_*z_))
x1x20x30.fill(0)
print("x1x20x30 shape: ",x1x20x30.shape)
around_point = []
min_point = []
min_dis = []
center_list = []
radius_list = []
process_list = [0,1,2,3,4,5,6,7,8,9]
print("save dir ", save_dir ,"begin")
# process_list = [1,3,6]
# process_list = [7]
# process_list = range(0,10)
with multiprocessing.Pool() as pool:
    pool.map(process_item, process_list)
    # pool.map(process_item, range(len(cdis_0_all_name)))
print(save_dir , " is over!!!!!!!!!!!!!!!!!!!1")
