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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from multiprocessing import Process, Manager
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
max_arg_max = []
#采样点数量初始化
x_ = 400
y_ = 400
z_ = 40
#范围
# x_r_max = 400
# x_r_min = -200
# y_r_max = 200
# y_r_min = -400
#坐标轴调转
x_r_max = -100
x_r_min = 400
y_r_max = -350
y_r_min = 200
z_r_max = 10
z_r_min = -10
#821 uwbinit
after1357_1 = [[44.03869330429116, 106.50327319126995, 10.950000131613828], 
               [153.26670106840223, 24.709319918748935, 7.879473443801533], 
               [232.93730724228448, -54.57772193957685, 6.672792034629818], 
               [331.09642919076737, -174.67498797502785, -0.10336786612246865], 
               [197.28934814524632, -206.64891824921014, -4.981101615333129], 
               [42.92717003799312, -218.5168637057965, -4.235250364817126], 
               [-16.521432197407506, -170.5434720151188, -1.9253234939963482], 
               [-52.360234396365485, -35.496824179331135, 9.667296670411831], 
               [43.4017932655219, -57.24202960465386, 1.8050967755580742], 
               [87.22705503788592, -25.922636462762608, 2.747934343232293]]

x10 = np.linspace(x_r_min, x_r_max, x_)
x20 = np.linspace(y_r_min, y_r_max, y_)
x30 = np.linspace(z_r_min, z_r_max, z_)
x1x20 = np.array([0]*(y_*x_))
x1x2x30 = np.array([0]*(y_*x_*z_))
x1x20.fill(0)
x1x2x30.fill(0)
# fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig, axs = plt.subplots(1, 2, figsize=(60, 40), dpi=150)
# 第一、二个子图
ax1 = axs[0]
ax1.set_title('anchor + Predict 1/sqrt(dis)')
ax1.set_xlim([-400, 500])
ax1.set_ylim([-400, 500])
ax2 = axs[1]
ax2.set_title('anchor + pre dis')
ax2.set_xlim([-400, 500])
ax2.set_ylim([-400, 500])
first = False
second = False
third = False
after_1357 = True
after_inv = True
#从第二个开始
cood_list = []
five_list = []
aveg_list = []
draw_list = [1,2,3,4,5,6,7,8,9]
# draw_list = [1,4,8,9]
# draw_list = range(1,10)
# for i in range(0,10):
# for i in draw_list:
# 定义每个线程的预测函数
def predict(chunk, chunk_indices):
    predicted_dis, predicted_sigma = pipeline.predict(chunk, return_std=True)
    return chunk_indices, predicted_dis, predicted_sigma

save_dir = "/home/lbd/slam/Lio/ntu/makern35/2/figure/3/draw_3/"
save_dir = "/home/lbd/slam/Lio/ntu/makern35/2/figure/3/3/sqrt_random_sample2/"
save_dir = "/home/lbd/slam/Lio/ntu/makern35/2/"
using_every_ep = True

for i in tqdm(draw_list, desc="Processing"):
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    print("using init uwb init!")
    print("time to draw anchor :", i)
    time_now = time.time()
    model_filename = save_dir + 'pipeline_model3D_sqrtinv' + str(i) + '.pkl'
    # 重新初始化图
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    fig, axs = plt.subplots(1, 2, figsize=(60, 40), dpi=150)
    # 第一、二个子图
    ax1 = axs[0]
    ax1.set_title('anchor + Predict 1/sqrt(dis)')
    ax1.set_xlim([-400, 500])
    ax1.set_ylim([-400, 500])
    ax2 = axs[1]
    ax2.set_title('anchor + pre dis')
    ax2.set_xlim([-400, 500])
    ax2.set_ylim([-400, 500])
    pipeline = joblib.load(model_filename)
    if(after1357_1):
        pos = after1357_1[i]
        # x_ = 200
        # y_ = 200
        # z_ = 2500
        #之前是10000
        # x_ = 600
        # y_ = 600
        # z_ = 300
        #//之前是1000 100 3
        x_ = 800
        y_ = 800
        z_ = 2
        dx = 3
        dy = 3
        dz = 0.05
        # max_x = np.linspace(pos[0] - dx, pos[0] + dx, x_)
        # max_y = np.linspace(pos[1] - dy, pos[1] + dy, y_)
        # max_z = np.linspace(pos[2] - dz, pos[2] + dz, z_)  
        max_x = np.linspace(-400, 500, x_)
        max_y = np.linspace(-400, 500, y_)
        max_z = np.linspace(pos[2] - dz, pos[2] + dz, z_)  
        # max_z = np.linspace(pos[2] - 10, pos[2] + 10, z_)  
    # 生成二维坐标点
    # init over 

    max_xx, max_yy = np.meshgrid(max_x, max_y)
    # p3 = np.array(list(product(max_x, max_y)))
    points = np.array(list(product(max_x, max_y, max_z)))
    print(points.shape)
    print("x y z: ", x_, y_, z_, "  dx dy dz : ",dx , dy, dz)
    # 设置每次处理的块大小
    num_threads = 20 #20
    num_to_predict = 500 #5000
    total_data = len(points)
    chunk_between = num_threads * num_to_predict
    chunk_size = total_data // (chunk_between)  # 减少每次处理的数据量

    # 使用多轮循环处理数据
    predicted_dis_list = []
    predicted_sigma_list = []
    start_idx = 0
    time_now = time.time()
    # 使用字典存储结果
    results_dict = {}

    print("开始多线程计算")

    # 多线程预测
    while start_idx < total_data:
        end_idx = min(start_idx + chunk_size * num_threads, total_data)
        
        # 分块数据和索引
        chunks = [points[i:i + chunk_size] for i in range(start_idx, end_idx, chunk_size)]
        chunk_indices = [np.arange(i, i + chunk_size) for i in range(start_idx, end_idx, chunk_size)]

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 提交任务到线程池
            futures = {executor.submit(predict, chunk, idx): idx for chunk, idx in zip(chunks, chunk_indices)}
            
            # 处理结果
            for future in as_completed(futures):
                try:
                    chunk_indices, predicted_dis, predicted_sigma = future.result()
                    
                    # 将结果存储到字典中，键为索引
                    for idx, dis, sigma in zip(chunk_indices, predicted_dis, predicted_sigma):
                        results_dict[idx] = (dis, sigma)
                except Exception as e:
                    print(f"Error in future: {e}")

        start_idx = end_idx

    print("多线程计算完成")
    print("time cost ", time.time() - time_now)
    # 打印 results_dict 的长度
    print("results_dict 的长度是:", len(results_dict))
    predicted_dis = []
    predicted_sigma = []

    # 遍历每个小块的索引，从0到num_threads * 500 - 1
    for t in range(total_data):
        dis, sigma = results_dict[t]
        predicted_dis_list.append(dis)
        predicted_sigma_list.append(sigma)

    # 将列表转换为 numpy 数组
    predicted_dis = np.array(predicted_dis_list)
    predicted_sigma = np.array(predicted_sigma_list)

    # 确保预测结果和 points 数量相同
    assert predicted_dis.shape[0] == points.shape[0]
    # predicted_dis = np.concatenate(predicted_dis_list)
    # predicted_sigma = np.concatenate(predicted_sigma_list)
    # 打印或使用结果
    print("predict res : ")
    print(predicted_dis.shape)
    print(predicted_sigma.shape)
    # 保存到文件
    np.save(save_dir + f'predicted_dis_{i}.npy', predicted_dis)
    np.save(save_dir + f'predicted_sigma_{i}.npy', predicted_sigma)

    print(f"Saved predicted_dis and predicted_sigma for i={i}")
    print("predicted over!")
    #取倒数作为可视化结果
    print("predicted_dis", predicted_dis.shape)
    print("x1x20", x1x20.shape)
    #不重构 直接画点
    max_value = np.max(predicted_dis)
    max_index = np.unravel_index(np.argmax(predicted_dis), predicted_dis.shape)
    cood = points[max_index]
    cood_list.append(cood)
    # 获取前10个最大值的索引
    flat_indices = np.argsort(predicted_dis.ravel())[-10:]  # 获取前10个最大值的扁平索引
    top10_values = predicted_dis.ravel()[flat_indices]  # 获取前10个最大值
    top10_indices = np.unravel_index(flat_indices, predicted_dis.shape)  # 转换为原数组的索引
    top10_cood = points[top10_indices]
    average_cood = np.mean(top10_cood, axis=0)
    aveg_list.append(average_cood)
    print("前10个最大值:", top10_values)
    print("平均值：",average_cood)
    print("对应的索引:", top10_indices)
    print("前10个最大值对应的点:", top10_cood)
    print("max cood : ", cood)
    # ###重构成x y z
    predicted_dis_re = predicted_dis.reshape(x_, y_, z_)
    #排平到2D
    # max_2 = np.max(predicted_dis_re[..., 2], axis=2)
    max_2 = np.max(predicted_dis_re, axis=2)
    # 遍历 x1x2x3 以找到每个 (x1, x2) 组合的最大第三维值

    # 排平到 2D
    x1x2 = max_2
    print("max x1x2",x1x2.shape)
    print("time to figure !")
    plt.axis('equal')
    ax = axs[0]
    ax.set_title('anchor + Predict '+str(i)+' 1/sqrt(dis)')
    # ax.set_title('anchor + Predict '+str(i)+' 1/dis')
    x_1 = np.linspace(x_r_min, x_r_max, x_+1)
    y_1 = np.linspace(y_r_min, y_r_max, y_+1)
    c1 = ax.pcolormesh( max_xx,max_yy, max_2.T, shading='auto', vmin=0, vmax=np.max(max_2))
    ax.scatter(pos[0], pos[1], color='g')
    if(i==9):
        fig.colorbar(c1, ax=ax)
    
    # 找到最大值及其索引
    max_value = np.max(predicted_dis_re)
    max_index = np.unravel_index(np.argmax(predicted_dis_re), predicted_dis_re.shape)

    print(f"最大值: {max_value}")
    print(f"对应的原值：{1 / (max_value**2)}")#{1 / (max_value)}")
    print(f"最大值的索引: {max_index}")
    print(f"对应的 x 值: {max_x[max_index[0]]}")
    print(f"对应的 y 值: {max_y[max_index[1]]}")
    print(f"对应的 z 值: {max_z[max_index[2]]}")
    print("cood is : ", cood)
    print("mean cood is : ",average_cood)
    ax.scatter([cood[0]], [cood[1]], color='r')
    # ax.scatter([max_x[max_index[0]]], [max_y[max_index[1]]], color='r')
    max_arg_max.append(max_index)
    # 打印最大值及其对应的坐标
    print("Maximum value:", max_value)
    print("Coordinates of maximum value:", max_index)
    for mi in max_arg_max:
        # 进一步找到实际坐标值（如果x1, x2代表的是坐标轴上的点）
        max_x1 = max_x[mi[0]]  # 因为 x 坐标对应数组的列索引
        max_x2 = max_y[mi[1]]  # 因为 y 坐标对应数组的行索引
        max_x3 = max_z[mi[2]]  # 因为 z?
        print( "Actual coordinates in the grid:", (max_x1, max_x2, max_x3))
        # 在图中标记最大值位置
        # ax.scatter([max_x1], [max_x2], color='r')
    # Predict ori mean subplot
    for mi in cood_list:
        print("cood is : ", (mi[0],mi[1],mi[2]))
    max_2_ori = 1 / ((max_2)**2 + 0.00000001) 
    # max_2_ori = 1 / ((max_2)) 
    ax = axs[1]
    ax.set_title('anchor + Predict '+str(i)+' dis')
    c2 = ax.pcolormesh(max_xx,max_yy, max_2_ori.T, shading='auto', vmin=0, vmax=np.max(max_2_ori))
    ax.scatter([cood[0]], [cood[1]], color='r')
    ax.scatter(pos[0], pos[1], color='g')
    # ax.scatter([max_x[max_index[0]]], [max_y[max_index[1]]], color='r')
    if(i==9):
        fig.colorbar(c2, ax=ax)
    for mi in max_arg_max:
        # 进一步找到实际坐标值（如果x1, x2代表的是坐标轴上的点）
        max_x1 = max_x[mi[0]]  # 因为 x 坐标对应数组的列索引
        max_x2 = max_y[mi[1]]  # 因为 y 坐标对应数组的行索引
        max_x3 = max_z[mi[2]]  # 因为 z?
        # print( "Actual coordinates in the grid:", (max_x1, max_x2, max_x3))
        # 在图中标记最大值位置
    plt.savefig(save_dir+ '/samplesqrt0911_init_'+str(i)+'.png')  # 保存为 output.png 文件

print("all cood:")
for mi in cood_list:
    print([mi[0],mi[1],mi[2]])
print("mean cood:")
for mi in aveg_list:
    print([mi[0],mi[1],mi[2]])
print("print all data")
print(after1357_1)
print("print all ten all!")
for m in range(len(five_list)):
    print("this is : ",m)
    print(five_list[m])

