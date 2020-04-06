# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:26:25 2020

@author: hxy
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)
import pygal


RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
x = list(RI_dict)
y =  list(RI_dict.values())
# plt.title("Matplotlib demo") 
plt.title(u'RI随机一致性指标', fontproperties=font_set)
plt.xlabel(u'随机矩阵阶数', fontproperties=font_set) 
plt.ylabel(u'随机一致指标值', fontproperties=font_set) 
plt.plot(x,y,"ob") 
plt.show()


def get_w(array):
    row = array.shape[0]  # 计算出阶数
    
    a_axis_0_sum = array.sum(axis=0)
    # print(a_axis_0_sum)
    b = array / a_axis_0_sum  # 新的矩阵b
    # print(b)
    b_axis_0_sum = b.sum(axis=0)
    b_axis_1_sum = b.sum(axis=1)  # 每一行的特征向量
    # print(b_axis_1_sum)
    w = b_axis_1_sum / row  # 归一化处理(特征向量)
    nw = w * row
    AW = (w * array).sum(axis=1)
    # print(AW)
    max_max = sum(AW / (row * w))
    # print(max_max)
    CI = (max_max - row) / (row - 1)
    CR = CI / RI_dict[row]
    if CR < 0.1  :
            print(round(CR, 3))
            print('满足一致性')
            print(np.max(w))
            print(sorted(w,reverse=True))
            print(max_max)
            print('特征向量:%s' % w)
            return w
    else:
        print(round(CR, 3))
        print('不满足一致性，请进行修改')
        
        print('若是非单层检验，则可以忽略一致性检验算法')
        print(np.max(w))
        print(sorted(w,reverse=True))
        print(max_max)
        print('特征向量:%s' % w)
        return w



def main(array):
    if type(array) is np.ndarray:
        return get_w(array)
    else:
        print('请输入numpy对象')


if __name__ == '__main__':
    # 由于地方问题，矩阵我就写成一行了
    # e = np.array([[1, 2, 7, 5, 5], [1 / 2, 1, 4, 3, 3], [1 / 7, 1 / 4, 1, 1 / 2, 1 / 3], [1 / 5, 1 / 3, 2, 1, 1], [1 / 5, 1 / 3, 3, 1, 1]])
    # a = np.array([[1, 1 / 3, 1 / 8], [3, 1, 1 / 3], [8, 3, 1]])
    # b = np.array([[1, 2, 5], [1 / 2, 1, 2], [1 / 5, 1 / 2, 1]])
    # c = np.array([[1, 1, 3], [1, 1, 3], [1 / 3, 1 / 3, 1]])
    # d = np.array([[1, 3, 4], [1 / 3, 1, 1], [1 / 4, 1, 1]])
    # f = np.array([[1, 4, 1 / 2], [1 / 4, 1, 1 / 4], [2, 4, 1]])
    
    # e = np.array([[1, 1/7, 1/3, 1/5], [7, 1, 1/5, 1/3], [3, 5, 1, 3], [ 5, 3, 1/3, 1]])
    # a = np.array([[1, 5, 4,6,2,9], [1/5,1,1/3,2,1/5,5], [1/4,3,1,4,1/4,5],[1/6,1/3,1/4,1,1/7,3],[1/2,5,4,7,1,8],[1/9,1/5,1/5,1/3,1/8,1]])
    # b = np.array([[1,1/2 , 1/2,1/3,1/2,1/2], [2,1,2,1,2,2], [2,1/2,1,1/2,1/2,1],[3,1,2,1,3,1/2],[2,1/2,2,1/3,1,1/2],[2,1/2,1,2,2,1]])
    # c = np.array([[1, 3, 5,4,7], [1/3,2,3,2,5], [1/5,1/3,1/2,1/2,3],[1/4,1/2,1,1,3],[1/7,1/5,1/3,1/3,1]])
    # d = np.array([[1, 2, 3,4,7], [1/2,1,3,2,5], [1/3,1/3,1,1/2,1],[1/4,1/2,2,1,3],[1/7,1/5,1,1/3,1]])
    
    
    
    
    # 绩效评估
    a = np.array([[1, 1 / 3, 1 / 8], [3, 1, 1 / 3], [8, 3, 1]])
    b = np.array([[1, 2, 5], [1 / 2, 1, 2], [1 / 5, 1 / 2, 1]])
    c = np.array([[1, 1, 3], [1, 1, 3], [1 / 3, 1 / 3, 1]])
    d = np.array([[1, 3, 4], [1 / 3, 1, 1], [1 / 4, 1, 1]])
    e = np.array([[1, 1/7, 1/3, 1/5], [7, 1, 1/5, 1/3], [3, 5, 1, 3], [ 5, 3, 1/3, 1]])
    
    e = main(e)
    a = main(a)
    b = main(b)
    c = main(c)
    d = main(d)
    # f = main(f)
    
    
    

 
    
     
    # x = range(5)
    # x_1=range(6)
    # x_2=range(4)
    
    # #plt.plot(x, y, 'ro-')
    # #plt.plot(x, y1, 'bo-')
    # #pl.xlim(-1, 11)  # 限定横轴的范围
    # #pl.ylim(-1, 110)  # 限定纵轴的范围
    # # plt.plot(x_1, list(a), marker='o', mec='r', mfc='w',label='politics weight')
    # # plt.plot(x_1, list(b), marker='*', ms=10,label='economic weight')
    # plt.plot(x_2, list(e), marker='o', mec='r', mfc='w',label='government weight')
    
     
     
    # # plt.plot(x, list(c), marker='o', mec='r', mfc='w',label='cultural weight')
    # # plt.plot(x, list(d), marker='*', ms=10,label='social weight')
    # plt.legend()  # 让图例生效
    # plt.xticks(x_1, rotation=1)
     
    # plt.margins(0)
    # plt.subplots_adjust(bottom=0.10)
    
    # plt.title(u'权重分布图', fontproperties=font_set)
    # plt.xlabel(u'准则层权重', fontproperties=font_set) 
    # plt.ylabel(u'权重数值', fontproperties=font_set)
    
    
    # plt.yticks([0.00,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00])
    # plt.show()
    # plt.title("A simple plot") #标题
    # plt.savefig('D:\\f1.jpg',dpi = 900)

    
    try:
        
        # res = np.array([a, b, c, d, f])
        res = np.array([a, b, c, d])
        

        
        print(res)
        
        ret = (np.transpose(res) * e).sum(axis=1)
        res_sum=np.array([a,b,c,d,ret])
        res_sum=res_sum.transpose()
        print(res_sum)
        print("这三个城市的排名：",ret)
        
        data=res_sum
        labels=['市政府A','市政府B','市政府C']
        radar = pygal.Radar()
        for i,per in enumerate(labels):

            radar.add(labels[i],data[i])
        radar.x_labels = ['政治绩效','经济绩效','文化绩效','社会绩效','总体得分']
        radar.dots_size = 8
        radar.legend_at_bottom = True
        radar.title = '城市权重以及总得分数据对比图'
        radar.render_to_file('goverment performance evaluation.svg')
        
        pass
    except TypeError:
        print('数据有误，可能不满足一致性，请进行修改')