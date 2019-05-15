# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 18:41:07 2019

@author: Administrator
"""

import math
import numpy as np 
import pylab as pl
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import fftpack  
import scipy.signal as signal
from scipy import interpolate
import random
# from keras.callbacks import Tensorboard #传说可以可视化------------------->????????????????
#from sklearn.cross_validation import train_test_split
import pandas as pd
from pylab import *
import os
np.random.seed(42)

#判定当前的时间序列是否是单调序列
def ismonotonic(x):
    max_peaks=signal.argrelextrema(x,np.greater)[0]
    min_peaks=signal.argrelextrema(x,np.less)[0]
    all_num=len(max_peaks)+len(min_peaks)
    if all_num>0:
        return False
    else:
        return True
        
#寻找当前时间序列的极值点
def findpeaks(x):
    
#     df_index=np.nonzero(np.diff((np.diff(x)>=0)+0)<0)
    
#     u_data=np.nonzero((x[df_index[0]+1]>x[df_index[0]]))
#     df_index[0][u_data[0]]+=1
    
#     return df_index[0]
    return signal.argrelextrema(x,np.greater)[0]
#判断当前的序列是否为 IMF 序列
def isImf(x):
    N=np.size(x)
    pass_zero=np.sum(x[0:N-2]*x[1:N-1]<0)#过零点的个数
    peaks_num=np.size(findpeaks(x))+np.size(findpeaks(-x))#极值点的个数
    if abs(pass_zero-peaks_num)>1:
        return False
    else:
        return True
#获取当前样条曲线
def getspline(x):
    N=np.size(x)
    peaks=findpeaks(x)
#     print '当前极值点个数：',len(peaks)
    peaks=np.concatenate(([0],peaks))
    peaks=np.concatenate((peaks,[N-1]))
    if(len(peaks)<=3):
#         if(len(peaks)<2):
#             peaks=np.concatenate(([0],peaks))
#             peaks=np.concatenate((peaks,[N-1]))
#             t=interpolate.splrep(peaks,y=x[peaks], w=None, xb=None, xe=None,k=len(peaks)-1)
#             return interpolate.splev(np.arange(N),t)
        t=interpolate.splrep(peaks,y=x[peaks], w=None, xb=None, xe=None,k=len(peaks)-1)
        return interpolate.splev(np.arange(N),t)
    t=interpolate.splrep(peaks,y=x[peaks])
    return interpolate.splev(np.arange(N),t)
#     f=interp1d(np.concatenate(([0,1],peaks,[N+1])),np.concatenate(([0,1],x[peaks],[0])),kind='cubic')
#     f=interp1d(peaks,x[peaks],kind='cubic')
#     return f(np.linspace(1,N,N))
    
    
#经验模态分解方法
def emd(x):
    imf=[]
    while not ismonotonic(x):
        x1=x
        sd=np.inf
        while sd>0.2 or  (not isImf(x1)):#===================================
#             print isImf(x1)
            s1=getspline(x1)
            s2=-getspline(-1*x1)
            x2=x1-(s1+s2)/2
            sd=np.sum((x1-x2)**2)/np.sum(x1**2)
            x1=x2
        
        imf.append(x1)
        x=x-x1
    imf.append(x)
    return imf
def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

def get_data_Fx(file,a,num):
    data_all= pd.read_csv(file,header = None,error_bad_lines=False)
    Force_x     = data_all[0]
    Force_x     = np.array(Force_x).reshape(1,-1)
    Force_x_score1 = (Force_x - np.mean(Force_x)) / np.std(Force_x)
    Force_x = Force_x_score1[:,a:a+num]
    return Force_x    # (1,10000)
def get_data_Fy(file,a,num):
    data_all= pd.read_csv(file,header = None,error_bad_lines=False)
    Force_y     = data_all[1]
    Force_y     = np.array(Force_y).reshape(1,-1)
    Force_y_score1 = (Force_y - np.mean(Force_y)) / np.std(Force_y)
    Force_y = Force_y_score1[:,a:a+num]
    return Force_y    # (1,10000)
def get_data_Fz(file,a,num):
    data_all= pd.read_csv(file,header = None,error_bad_lines=False)
    Force_z     = data_all[2]
    Force_z     = np.array(Force_z).reshape(1,-1)
    Force_z_score1 = (Force_z - np.mean(Force_z)) / np.std(Force_z)
    Force_z = Force_z_score1[:,a:a+num]
    return Force_z    # (1,10000)
def get_data_Vx(file,a,num):
    data_all= pd.read_csv(file,header = None,error_bad_lines=False)
    vibration_x     = data_all[0]
    vibration_x     = np.array(vibration_x).reshape(1,-1)
    vibration_x_score1 = (vibration_x - np.mean(vibration_x)) / np.std(vibration_x)
    vibration_x = vibration_x_score1[:,a:a+num]
    return vibration_x    # (1,10000)
def get_data_Vy(file,a,num):
    data_all= pd.read_csv(file,header = None,error_bad_lines=False)
    vibration_y     = data_all[0]
    vibration_y     = np.array(vibration_y).reshape(1,-1)
    vibration_y_score1 = (vibration_y - np.mean(vibration_y)) / np.std(vibration_y)
    vibration_y = vibration_y_score1[:,a:a+num]
    return vibration_y    # (1,10000)
def get_data_Vz(file,a,num):
    data_all= pd.read_csv(file,header = None,error_bad_lines=False)
    vibration_z     = data_all[0]
    vibration_z     = np.array(vibration_z).reshape(1,-1)
    vibration_z_score1 = (vibration_z - np.mean(vibration_z)) / np.std(vibration_z)
    vibration_z = vibration_z_score1[:,a:a+num]
    return vibration_z    # (1,10000)

a=30000
num=5000

os.chdir(r'H:\industry-big-data\PHM2010\c1')
filename_list1=[]
filename_list1=os.listdir(r'H:\industry-big-data\PHM2010\c1\c1')
print(len(filename_list1))
y1_max = pd.read_csv(filepath_or_buffer = r'H:\industry-big-data\PHM2010\c1\c1_wear.csv', sep = ',')["max"].values
y1_mean = pd.read_csv(filepath_or_buffer = r'H:\industry-big-data\PHM2010\c1\c1_wear.csv', sep = ',')["mean"].values
print(y1_max.shape,'y1_max',y1_mean.shape,'y1_mean')


os.chdir(r'H:\industry-big-data\PHM2010\c1\c1')
i=0
x1=[]
x1=np.array(x1)
x1 = x1.reshape(-1, num)
for name in filename_list1:
    X_linshi = get_data_Fx(name,a,num)
    x1 = np.vstack((x1,X_linshi))
    i=i+1
print('x1.shape:',x1.shape,i)

#os.chdir(r'H:\industry-big-data\PHM2010\c1\c1')
#i=0
#y1=[]
#y1=np.array(y1)
#y1 = y1.reshape(-1, num)
#for name in filename_list1:
#    Y_linshi = get_data_Fx(name,a,num)
#    y1 = np.vstack((y1,Y_linshi))
#    i=i+1
#print('y1.shape:',y1.shape,i)
#
#os.chdir(r'H:\industry-big-data\PHM2010\c1\c1')
#i=0
#z1=[]
#z1=np.array(z1)
#z1 = z1.reshape(-1, num)
#for name in filename_list1:
#    Z_linshi = get_data_Fx(name,a,num)
#    z1 = np.vstack((z1,Z_linshi))
#    i=i+1
#print('z1.shape:',z1.shape,i)
#
#sampling_rate=50000
#t=np.arange(0, 0.1, 1.0/sampling_rate)

#imf_x0=emd(x1[0])
#save = pd.DataFrame(imf_x0)
#save = save.T
#save.to_csv('imf_x0.csv',index=False,header=False)
#os.chdir(r'H:\python')
#for i in range(10):
#    imf =emd(x1[i])
#    save = pd.DataFrame(imf)
#    save = save.T
#    save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
os.chdir(r'H:\python')
i=1
imf1 =emd(x1[i])
save = pd.DataFrame(imf1)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=2
imf2 =emd(x1[i])
save = pd.DataFrame(imf2)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=3
imf3 =emd(x1[i])
save = pd.DataFrame(imf3)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=4
imf4 =emd(x1[i])
save = pd.DataFrame(imf4)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=5
imf5 =emd(x1[i])
save = pd.DataFrame(imf5)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=6
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=7
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=8
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=9
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=10
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=11
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=12
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=13
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=14
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=15
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=16
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=17
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=18
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=19
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=20
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=21
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=22
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=23
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=24
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=25
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=26
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=27
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=28
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=29
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=30
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=31
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=32
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=33
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=34
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=35
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=36
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=37
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=38
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=39
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=40
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=41
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=42
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=43
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=44
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=45
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=46
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=47
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=48
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=49
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=50
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=51
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=52
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=53
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=54
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=55
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=56
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=57
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=58
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=59
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=60
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=61
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=62
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=63
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=64
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=65
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=66
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=67
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=68
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=69
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=70
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=71
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=72
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=73
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=74
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=75
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=76
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=77
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=78
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=79
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=80
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=81
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=82
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=83
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=84
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=85
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=86
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=87
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=88
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=89
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=90
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=91
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=92
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=93
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=94
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=95
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=96
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=97
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=98
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=99
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=100
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=101
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=102
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=103
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=104
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=105
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=106
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=107
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=108
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=109
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=110
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=111
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=112
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=113
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=114
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=115
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=116
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=117
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=118
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=119
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=120
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=121
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=122
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=123
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=124
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=125
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=126
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=127
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=128
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=129
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=130
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=131
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=132
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=133
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=134
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=135
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=136
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=137
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=138
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=139
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=140
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=141
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=142
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=143
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=144
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=145
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=146
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=147
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=148
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=149
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=150
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=151
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=152
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=153
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=154
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=155
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=156
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=157
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=158
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=159
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=160
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=161
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=162
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=163
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=164
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=165
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=166
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=167
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=168
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=169
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=170
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=171
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=172
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=173
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=174
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=175
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=176
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=177
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=178
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=179
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=180
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=181
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=182
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=183
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=184
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=185
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=186
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=187
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=188
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=189
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=190
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=191
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=192
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=193
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=194
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=195
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=196
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=197
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=198
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=199
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=200
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=201
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=202
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=203
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=204
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=205
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=206
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=207
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=208
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=209
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=210
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=211
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=212
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=213
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=214
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=215
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=216
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=217
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=218
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=219
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=220
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=221
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=222
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=223
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=224
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=225
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=226
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=227
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=228
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=229
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=230
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=231
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=232
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=233
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=234
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=235
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=236
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=237
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=238
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=239
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=240
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=241
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=242
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=243
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=244
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=245
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=246
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=247
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=248
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=249
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=250
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=251
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=252
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=253
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=254
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=255
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=256
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=257
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=258
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=259
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=260
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=261
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=262
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=263
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=264
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=265
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=266
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=267
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=268
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=269
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=270
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=271
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=272
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=273
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=274
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=275
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=276
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=277
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=278
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=279
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=280
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=281
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=282
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=283
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=284
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=285
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=286
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=287
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=288
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=289
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=290
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=291
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=292
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=293
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=294
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=295
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=296
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=297
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=298
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=299
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=300
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=301
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=302
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=303
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=304
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=305
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=306
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=307
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=308
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=309
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=310
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=311
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=312
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=313
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=314
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
i=315
imf =emd(x1[i])
save = pd.DataFrame(imf)
save = save.T
save.to_csv('imf_x%s.csv'%(i),index=False,header=False)

#i=9
#while i<11:
#    imf =emd(x1[i])
#    save = pd.DataFrame(imf)
#    save = save.T
#    save.to_csv('imf_x%s.csv'%(i),index=False,header=False)
#    i=i+1
#imf_x1=emd(x1[1])
#imf_x2=emd(x1[2])
#imf_x3=emd(x1[3])
#imf_x4=emd(x1[4])
#imf_x5=emd(x1[5])
#imf_x6=emd(x1[6])
#imf_x7=emd(x1[7])
#imf_x8=emd(x1[8])
#imf_x9=emd(x1[9])
#imf_x10=emd(x1[10])
#imf_x11=emd(x1[11])
#imf_x12=emd(x1[12])
#imf_x13=emd(x1[13])
#imf_x14=emd(x1[14])
#imf_x15=emd(x1[15])
#imf_x16=emd(x1[16])
#imf_x17=emd(x1[17])
#imf_x18=emd(x1[18])
#imf_x19=emd(x1[19])
#imf_x20=emd(x1[20])
#imf_x21=emd(x1[21])
#imf_x22=emd(x1[22])
#imf_x23=emd(x1[23])
#imf_x24=emd(x1[24])
#imf_x25=emd(x1[25])
#imf_x26=emd(x1[26])
#imf_x27=emd(x1[27])
#imf_x28=emd(x1[28])
#imf_x29=emd(x1[29])
#imf_x30=emd(x1[30])
#imf_x31=emd(x1[31])
#imf_x32=emd(x1[32])
#imf_x33=emd(x1[33])
#imf_x34=emd(x1[34])
#imf_x35=emd(x1[35])
#imf_x36=emd(x1[36])
#imf_x37=emd(x1[37])
#imf_x38=emd(x1[38])
#imf_x39=emd(x1[39])
#imf_x40=emd(x1[40])
#imf_x41=emd(x1[41])
#imf_x42=emd(x1[42])
#imf_x43=emd(x1[43])
#imf_x44=emd(x1[44])
#imf_x45=emd(x1[45])
#imf_x46=emd(x1[46])
#imf_x47=emd(x1[47])
#imf_x48=emd(x1[48])
#imf_x49=emd(x1[49])
#imf_x50=emd(x1[50])
#imf_x51=emd(x1[51])
#imf_x52=emd(x1[52])
#imf_x53=emd(x1[53])
#imf_x54=emd(x1[54])
#imf_x55=emd(x1[55])
#imf_x56=emd(x1[56])
#imf_x57=emd(x1[57])
#imf_x58=emd(x1[58])
#imf_x59=emd(x1[59])
#imf_x60=emd(x1[60])
#imf_x61=emd(x1[61])
#imf_x62=emd(x1[62])
#imf_x63=emd(x1[63])
#imf_x64=emd(x1[64])
#imf_x65=emd(x1[65])
#imf_x66=emd(x1[66])
#imf_x67=emd(x1[67])
#imf_x68=emd(x1[68])
#imf_x69=emd(x1[69])
#imf_x70=emd(x1[70])
#imf_x71=emd(x1[71])
#imf_x72=emd(x1[72])
#imf_x73=emd(x1[73])
#imf_x74=emd(x1[74])
#imf_x75=emd(x1[75])
#imf_x76=emd(x1[76])
#imf_x77=emd(x1[77])
#imf_x78=emd(x1[78])
#imf_x79=emd(x1[79])
#imf_x80=emd(x1[80])
#imf_x81=emd(x1[81])
#imf_x82=emd(x1[82])
#imf_x83=emd(x1[83])
#imf_x84=emd(x1[84])
#imf_x85=emd(x1[85])
#imf_x86=emd(x1[86])
#imf_x87=emd(x1[87])
#imf_x88=emd(x1[88])
#imf_x89=emd(x1[89])
#imf_x90=emd(x1[90])
#imf_x91=emd(x1[91])
#imf_x92=emd(x1[92])
#imf_x93=emd(x1[93])
#imf_x94=emd(x1[94])
#imf_x95=emd(x1[95])
#imf_x96=emd(x1[96])
#imf_x97=emd(x1[97])
#imf_x98=emd(x1[98])
#imf_x99=emd(x1[99])
#imf_x100=emd(x1[100])
#imf_x101=emd(x1[101])
#imf_x102=emd(x1[102])
#imf_x103=emd(x1[103])
#imf_x104=emd(x1[104])
#imf_x105=emd(x1[105])
#imf_x106=emd(x1[106])
#imf_x107=emd(x1[107])
#imf_x108=emd(x1[108])
#imf_x109=emd(x1[109])
#imf_x110=emd(x1[110])
#imf_x111=emd(x1[111])
#imf_x112=emd(x1[112])
#imf_x113=emd(x1[113])
#imf_x114=emd(x1[114])
#imf_x115=emd(x1[115])
#imf_x116=emd(x1[116])
#imf_x117=emd(x1[117])
#imf_x118=emd(x1[118])
#imf_x119=emd(x1[119])
#imf_x120=emd(x1[120])
#imf_x121=emd(x1[121])
#imf_x122=emd(x1[122])
#imf_x123=emd(x1[123])
#imf_x124=emd(x1[124])
#imf_x125=emd(x1[125])
#imf_x126=emd(x1[126])
#imf_x127=emd(x1[127])
#imf_x128=emd(x1[128])
#imf_x129=emd(x1[129])
#imf_x130=emd(x1[130])
#imf_x131=emd(x1[131])
#imf_x132=emd(x1[132])
#imf_x133=emd(x1[133])
#imf_x134=emd(x1[134])
#imf_x135=emd(x1[135])
#imf_x136=emd(x1[136])
#imf_x137=emd(x1[137])
#imf_x138=emd(x1[138])
#imf_x139=emd(x1[139])
#imf_x140=emd(x1[140])
#imf_x141=emd(x1[141])
#imf_x142=emd(x1[142])
#imf_x143=emd(x1[143])
#imf_x144=emd(x1[144])
#imf_x145=emd(x1[145])
#imf_x146=emd(x1[146])
#imf_x147=emd(x1[147])
#imf_x148=emd(x1[148])
#imf_x149=emd(x1[149])
#imf_x150=emd(x1[150])
#imf_x151=emd(x1[151])
#imf_x152=emd(x1[152])
#imf_x153=emd(x1[153])
#imf_x154=emd(x1[154])
#imf_x155=emd(x1[155])
#imf_x156=emd(x1[156])
#imf_x157=emd(x1[157])
#imf_x158=emd(x1[158])
#imf_x159=emd(x1[159])
#imf_x160=emd(x1[160])
#imf_x161=emd(x1[161])
#imf_x162=emd(x1[162])
#imf_x163=emd(x1[163])
#imf_x164=emd(x1[164])
#imf_x165=emd(x1[165])
#imf_x166=emd(x1[166])
#imf_x167=emd(x1[167])
#imf_x168=emd(x1[168])
#imf_x169=emd(x1[169])
#imf_x170=emd(x1[170])
#imf_x171=emd(x1[171])
#imf_x172=emd(x1[172])
#imf_x173=emd(x1[173])
#imf_x174=emd(x1[174])
#imf_x175=emd(x1[175])
#imf_x176=emd(x1[176])
#imf_x177=emd(x1[177])
#imf_x178=emd(x1[178])
#imf_x179=emd(x1[179])
#imf_x180=emd(x1[180])
#imf_x181=emd(x1[181])
#imf_x182=emd(x1[182])
#imf_x183=emd(x1[183])
#imf_x184=emd(x1[184])
#imf_x185=emd(x1[185])
#imf_x186=emd(x1[186])
#imf_x187=emd(x1[187])
#imf_x188=emd(x1[188])
#imf_x189=emd(x1[189])
#imf_x190=emd(x1[190])
#imf_x191=emd(x1[191])
#imf_x192=emd(x1[192])
#imf_x193=emd(x1[193])
#imf_x194=emd(x1[194])
#imf_x195=emd(x1[195])
#imf_x196=emd(x1[196])
#imf_x197=emd(x1[197])
#imf_x198=emd(x1[198])
#imf_x199=emd(x1[199])
#imf_x200=emd(x1[200])
#imf_x201=emd(x1[201])
#imf_x202=emd(x1[202])
#imf_x203=emd(x1[203])
#imf_x204=emd(x1[204])
#imf_x205=emd(x1[205])
#imf_x206=emd(x1[206])
#imf_x207=emd(x1[207])
#imf_x208=emd(x1[208])
#imf_x209=emd(x1[209])
#imf_x210=emd(x1[210])
#imf_x211=emd(x1[211])
#imf_x212=emd(x1[212])
#imf_x213=emd(x1[213])
#imf_x214=emd(x1[214])
#imf_x215=emd(x1[215])
#imf_x216=emd(x1[216])
#imf_x217=emd(x1[217])
#imf_x218=emd(x1[218])
#imf_x219=emd(x1[219])
#imf_x220=emd(x1[220])
#imf_x221=emd(x1[221])
#imf_x222=emd(x1[222])
#imf_x223=emd(x1[223])
#imf_x224=emd(x1[224])
#imf_x225=emd(x1[225])
#imf_x226=emd(x1[226])
#imf_x227=emd(x1[227])
#imf_x228=emd(x1[228])
#imf_x229=emd(x1[229])
#imf_x230=emd(x1[230])
#imf_x231=emd(x1[231])
#imf_x232=emd(x1[232])
#imf_x233=emd(x1[233])
#imf_x234=emd(x1[234])
#imf_x235=emd(x1[235])
#imf_x236=emd(x1[236])
#imf_x237=emd(x1[237])
#imf_x238=emd(x1[238])
#imf_x239=emd(x1[239])
#imf_x240=emd(x1[240])
#imf_x241=emd(x1[241])
#imf_x242=emd(x1[242])
#imf_x243=emd(x1[243])
#imf_x244=emd(x1[244])
#imf_x245=emd(x1[245])
#imf_x246=emd(x1[246])
#imf_x247=emd(x1[247])
#imf_x248=emd(x1[248])
#imf_x249=emd(x1[249])
#imf_x250=emd(x1[250])
#imf_x251=emd(x1[251])
#imf_x252=emd(x1[252])
#imf_x253=emd(x1[253])
#imf_x254=emd(x1[254])
#imf_x255=emd(x1[255])
#imf_x256=emd(x1[256])
#imf_x257=emd(x1[257])
#imf_x258=emd(x1[258])
#imf_x259=emd(x1[259])
#imf_x260=emd(x1[260])
#imf_x261=emd(x1[261])
#imf_x262=emd(x1[262])
#imf_x263=emd(x1[263])
#imf_x264=emd(x1[264])
#imf_x265=emd(x1[265])
#imf_x266=emd(x1[266])
#imf_x267=emd(x1[267])
#imf_x268=emd(x1[268])
#imf_x269=emd(x1[269])
#imf_x270=emd(x1[270])
#imf_x271=emd(x1[271])
#imf_x272=emd(x1[272])
#imf_x273=emd(x1[273])
#imf_x274=emd(x1[274])
#imf_x275=emd(x1[275])
#imf_x276=emd(x1[276])
#imf_x277=emd(x1[277])
#imf_x278=emd(x1[278])
#imf_x279=emd(x1[279])
#imf_x280=emd(x1[280])
#imf_x281=emd(x1[281])
#imf_x282=emd(x1[282])
#imf_x283=emd(x1[283])
#imf_x284=emd(x1[284])
#imf_x285=emd(x1[285])
#imf_x286=emd(x1[286])
#imf_x287=emd(x1[287])
#imf_x288=emd(x1[288])
#imf_x289=emd(x1[289])
#imf_x290=emd(x1[290])
#imf_x291=emd(x1[291])
#imf_x292=emd(x1[292])
#imf_x293=emd(x1[293])
#imf_x294=emd(x1[294])
#imf_x295=emd(x1[295])
#imf_x296=emd(x1[296])
#imf_x297=emd(x1[297])
#imf_x298=emd(x1[298])
#imf_x299=emd(x1[299])
#imf_x300=emd(x1[300])
#imf_x301=emd(x1[301])
#imf_x302=emd(x1[302])
#imf_x303=emd(x1[303])
#imf_x304=emd(x1[304])
#imf_x305=emd(x1[305])
#imf_x306=emd(x1[306])
#imf_x307=emd(x1[307])
#imf_x308=emd(x1[308])
#imf_x309=emd(x1[309])
#imf_x310=emd(x1[310])
#imf_x311=emd(x1[311])
#imf_x312=emd(x1[312])
#imf_x313=emd(x1[313])
#imf_x314=emd(x1[314])
#print('imf_x_finish')
#imf_y0=emd(y1[0])
#imf_y1=emd(y1[1])
#imf_y2=emd(y1[2])
#imf_y3=emd(y1[3])
#imf_y4=emd(y1[4])
#imf_y5=emd(y1[5])
#imf_y6=emd(y1[6])
#imf_y7=emd(y1[7])
#imf_y8=emd(y1[8])
#imf_y9=emd(y1[9])
#imf_y10=emd(y1[10])
#imf_y11=emd(y1[11])
#imf_y12=emd(y1[12])
#imf_y13=emd(y1[13])
#imf_y14=emd(y1[14])
#imf_y15=emd(y1[15])
#imf_y16=emd(y1[16])
#imf_y17=emd(y1[17])
#imf_y18=emd(y1[18])
#imf_y19=emd(y1[19])
#imf_y20=emd(y1[20])
#imf_y21=emd(y1[21])
#imf_y22=emd(y1[22])
#imf_y23=emd(y1[23])
#imf_y24=emd(y1[24])
#imf_y25=emd(y1[25])
#imf_y26=emd(y1[26])
#imf_y27=emd(y1[27])
#imf_y28=emd(y1[28])
#imf_y29=emd(y1[29])
#imf_y30=emd(y1[30])
#imf_y31=emd(y1[31])
#imf_y32=emd(y1[32])
#imf_y33=emd(y1[33])
#imf_y34=emd(y1[34])
#imf_y35=emd(y1[35])
#imf_y36=emd(y1[36])
#imf_y37=emd(y1[37])
#imf_y38=emd(y1[38])
#imf_y39=emd(y1[39])
#imf_y40=emd(y1[40])
#imf_y41=emd(y1[41])
#imf_y42=emd(y1[42])
#imf_y43=emd(y1[43])
#imf_y44=emd(y1[44])
#imf_y45=emd(y1[45])
#imf_y46=emd(y1[46])
#imf_y47=emd(y1[47])
#imf_y48=emd(y1[48])
#imf_y49=emd(y1[49])
#imf_y50=emd(y1[50])
#imf_y51=emd(y1[51])
#imf_y52=emd(y1[52])
#imf_y53=emd(y1[53])
#imf_y54=emd(y1[54])
#imf_y55=emd(y1[55])
#imf_y56=emd(y1[56])
#imf_y57=emd(y1[57])
#imf_y58=emd(y1[58])
#imf_y59=emd(y1[59])
#imf_y60=emd(y1[60])
#imf_y61=emd(y1[61])
#imf_y62=emd(y1[62])
#imf_y63=emd(y1[63])
#imf_y64=emd(y1[64])
#imf_y65=emd(y1[65])
#imf_y66=emd(y1[66])
#imf_y67=emd(y1[67])
#imf_y68=emd(y1[68])
#imf_y69=emd(y1[69])
#imf_y70=emd(y1[70])
#imf_y71=emd(y1[71])
#imf_y72=emd(y1[72])
#imf_y73=emd(y1[73])
#imf_y74=emd(y1[74])
#imf_y75=emd(y1[75])
#imf_y76=emd(y1[76])
#imf_y77=emd(y1[77])
#imf_y78=emd(y1[78])
#imf_y79=emd(y1[79])
#imf_y80=emd(y1[80])
#imf_y81=emd(y1[81])
#imf_y82=emd(y1[82])
#imf_y83=emd(y1[83])
#imf_y84=emd(y1[84])
#imf_y85=emd(y1[85])
#imf_y86=emd(y1[86])
#imf_y87=emd(y1[87])
#imf_y88=emd(y1[88])
#imf_y89=emd(y1[89])
#imf_y90=emd(y1[90])
#imf_y91=emd(y1[91])
#imf_y92=emd(y1[92])
#imf_y93=emd(y1[93])
#imf_y94=emd(y1[94])
#imf_y95=emd(y1[95])
#imf_y96=emd(y1[96])
#imf_y97=emd(y1[97])
#imf_y98=emd(y1[98])
#imf_y99=emd(y1[99])
#imf_y100=emd(y1[100])
#imf_y101=emd(y1[101])
#imf_y102=emd(y1[102])
#imf_y103=emd(y1[103])
#imf_y104=emd(y1[104])
#imf_y105=emd(y1[105])
#imf_y106=emd(y1[106])
#imf_y107=emd(y1[107])
#imf_y108=emd(y1[108])
#imf_y109=emd(y1[109])
#imf_y110=emd(y1[110])
#imf_y111=emd(y1[111])
#imf_y112=emd(y1[112])
#imf_y113=emd(y1[113])
#imf_y114=emd(y1[114])
#imf_y115=emd(y1[115])
#imf_y116=emd(y1[116])
#imf_y117=emd(y1[117])
#imf_y118=emd(y1[118])
#imf_y119=emd(y1[119])
#imf_y120=emd(y1[120])
#imf_y121=emd(y1[121])
#imf_y122=emd(y1[122])
#imf_y123=emd(y1[123])
#imf_y124=emd(y1[124])
#imf_y125=emd(y1[125])
#imf_y126=emd(y1[126])
#imf_y127=emd(y1[127])
#imf_y128=emd(y1[128])
#imf_y129=emd(y1[129])
#imf_y130=emd(y1[130])
#imf_y131=emd(y1[131])
#imf_y132=emd(y1[132])
#imf_y133=emd(y1[133])
#imf_y134=emd(y1[134])
#imf_y135=emd(y1[135])
#imf_y136=emd(y1[136])
#imf_y137=emd(y1[137])
#imf_y138=emd(y1[138])
#imf_y139=emd(y1[139])
#imf_y140=emd(y1[140])
#imf_y141=emd(y1[141])
#imf_y142=emd(y1[142])
#imf_y143=emd(y1[143])
#imf_y144=emd(y1[144])
#imf_y145=emd(y1[145])
#imf_y146=emd(y1[146])
#imf_y147=emd(y1[147])
#imf_y148=emd(y1[148])
#imf_y149=emd(y1[149])
#imf_y150=emd(y1[150])
#imf_y151=emd(y1[151])
#imf_y152=emd(y1[152])
#imf_y153=emd(y1[153])
#imf_y154=emd(y1[154])
#imf_y155=emd(y1[155])
#imf_y156=emd(y1[156])
#imf_y157=emd(y1[157])
#imf_y158=emd(y1[158])
#imf_y159=emd(y1[159])
#imf_y160=emd(y1[160])
#imf_y161=emd(y1[161])
#imf_y162=emd(y1[162])
#imf_y163=emd(y1[163])
#imf_y164=emd(y1[164])
#imf_y165=emd(y1[165])
#imf_y166=emd(y1[166])
#imf_y167=emd(y1[167])
#imf_y168=emd(y1[168])
#imf_y169=emd(y1[169])
#imf_y170=emd(y1[170])
#imf_y171=emd(y1[171])
#imf_y172=emd(y1[172])
#imf_y173=emd(y1[173])
#imf_y174=emd(y1[174])
#imf_y175=emd(y1[175])
#imf_y176=emd(y1[176])
#imf_y177=emd(y1[177])
#imf_y178=emd(y1[178])
#imf_y179=emd(y1[179])
#imf_y180=emd(y1[180])
#imf_y181=emd(y1[181])
#imf_y182=emd(y1[182])
#imf_y183=emd(y1[183])
#imf_y184=emd(y1[184])
#imf_y185=emd(y1[185])
#imf_y186=emd(y1[186])
#imf_y187=emd(y1[187])
#imf_y188=emd(y1[188])
#imf_y189=emd(y1[189])
#imf_y190=emd(y1[190])
#imf_y191=emd(y1[191])
#imf_y192=emd(y1[192])
#imf_y193=emd(y1[193])
#imf_y194=emd(y1[194])
#imf_y195=emd(y1[195])
#imf_y196=emd(y1[196])
#imf_y197=emd(y1[197])
#imf_y198=emd(y1[198])
#imf_y199=emd(y1[199])
#imf_y200=emd(y1[200])
#imf_y201=emd(y1[201])
#imf_y202=emd(y1[202])
#imf_y203=emd(y1[203])
#imf_y204=emd(y1[204])
#imf_y205=emd(y1[205])
#imf_y206=emd(y1[206])
#imf_y207=emd(y1[207])
#imf_y208=emd(y1[208])
#imf_y209=emd(y1[209])
#imf_y210=emd(y1[210])
#imf_y211=emd(y1[211])
#imf_y212=emd(y1[212])
#imf_y213=emd(y1[213])
#imf_y214=emd(y1[214])
#imf_y215=emd(y1[215])
#imf_y216=emd(y1[216])
#imf_y217=emd(y1[217])
#imf_y218=emd(y1[218])
#imf_y219=emd(y1[219])
#imf_y220=emd(y1[220])
#imf_y221=emd(y1[221])
#imf_y222=emd(y1[222])
#imf_y223=emd(y1[223])
#imf_y224=emd(y1[224])
#imf_y225=emd(y1[225])
#imf_y226=emd(y1[226])
#imf_y227=emd(y1[227])
#imf_y228=emd(y1[228])
#imf_y229=emd(y1[229])
#imf_y230=emd(y1[230])
#imf_y231=emd(y1[231])
#imf_y232=emd(y1[232])
#imf_y233=emd(y1[233])
#imf_y234=emd(y1[234])
#imf_y235=emd(y1[235])
#imf_y236=emd(y1[236])
#imf_y237=emd(y1[237])
#imf_y238=emd(y1[238])
#imf_y239=emd(y1[239])
#imf_y240=emd(y1[240])
#imf_y241=emd(y1[241])
#imf_y242=emd(y1[242])
#imf_y243=emd(y1[243])
#imf_y244=emd(y1[244])
#imf_y245=emd(y1[245])
#imf_y246=emd(y1[246])
#imf_y247=emd(y1[247])
#imf_y248=emd(y1[248])
#imf_y249=emd(y1[249])
#imf_y250=emd(y1[250])
#imf_y251=emd(y1[251])
#imf_y252=emd(y1[252])
#imf_y253=emd(y1[253])
#imf_y254=emd(y1[254])
#imf_y255=emd(y1[255])
#imf_y256=emd(y1[256])
#imf_y257=emd(y1[257])
#imf_y258=emd(y1[258])
#imf_y259=emd(y1[259])
#imf_y260=emd(y1[260])
#imf_y261=emd(y1[261])
#imf_y262=emd(y1[262])
#imf_y263=emd(y1[263])
#imf_y264=emd(y1[264])
#imf_y265=emd(y1[265])
#imf_y266=emd(y1[266])
#imf_y267=emd(y1[267])
#imf_y268=emd(y1[268])
#imf_y269=emd(y1[269])
#imf_y270=emd(y1[270])
#imf_y271=emd(y1[271])
#imf_y272=emd(y1[272])
#imf_y273=emd(y1[273])
#imf_y274=emd(y1[274])
#imf_y275=emd(y1[275])
#imf_y276=emd(y1[276])
#imf_y277=emd(y1[277])
#imf_y278=emd(y1[278])
#imf_y279=emd(y1[279])
#imf_y280=emd(y1[280])
#imf_y281=emd(y1[281])
#imf_y282=emd(y1[282])
#imf_y283=emd(y1[283])
#imf_y284=emd(y1[284])
#imf_y285=emd(y1[285])
#imf_y286=emd(y1[286])
#imf_y287=emd(y1[287])
#imf_y288=emd(y1[288])
#imf_y289=emd(y1[289])
#imf_y290=emd(y1[290])
#imf_y291=emd(y1[291])
#imf_y292=emd(y1[292])
#imf_y293=emd(y1[293])
#imf_y294=emd(y1[294])
#imf_y295=emd(y1[295])
#imf_y296=emd(y1[296])
#imf_y297=emd(y1[297])
#imf_y298=emd(y1[298])
#imf_y299=emd(y1[299])
#imf_y300=emd(y1[300])
#imf_y301=emd(y1[301])
#imf_y302=emd(y1[302])
#imf_y303=emd(y1[303])
#imf_y304=emd(y1[304])
#imf_y305=emd(y1[305])
#imf_y306=emd(y1[306])
#imf_y307=emd(y1[307])
#imf_y308=emd(y1[308])
#imf_y309=emd(y1[309])
#imf_y310=emd(y1[310])
#imf_y311=emd(y1[311])
#imf_y312=emd(y1[312])
#imf_y313=emd(y1[313])
#imf_y314=emd(y1[314])
#print('imf_y_finish')
#imf_z0=emd(z1[0])
#imf_z1=emd(z1[1])
#imf_z2=emd(z1[2])
#imf_z3=emd(z1[3])
#imf_z4=emd(z1[4])
#imf_z5=emd(z1[5])
#imf_z6=emd(z1[6])
#imf_z7=emd(z1[7])
#imf_z8=emd(z1[8])
#imf_z9=emd(z1[9])
#imf_z10=emd(z1[10])
#imf_z11=emd(z1[11])
#imf_z12=emd(z1[12])
#imf_z13=emd(z1[13])
#imf_z14=emd(z1[14])
#imf_z15=emd(z1[15])
#imf_z16=emd(z1[16])
#imf_z17=emd(z1[17])
#imf_z18=emd(z1[18])
#imf_z19=emd(z1[19])
#imf_z20=emd(z1[20])
#imf_z21=emd(z1[21])
#imf_z22=emd(z1[22])
#imf_z23=emd(z1[23])
#imf_z24=emd(z1[24])
#imf_z25=emd(z1[25])
#imf_z26=emd(z1[26])
#imf_z27=emd(z1[27])
#imf_z28=emd(z1[28])
#imf_z29=emd(z1[29])
#imf_z30=emd(z1[30])
#imf_z31=emd(z1[31])
#imf_z32=emd(z1[32])
#imf_z33=emd(z1[33])
#imf_z34=emd(z1[34])
#imf_z35=emd(z1[35])
#imf_z36=emd(z1[36])
#imf_z37=emd(z1[37])
#imf_z38=emd(z1[38])
#imf_z39=emd(z1[39])
#imf_z40=emd(z1[40])
#imf_z41=emd(z1[41])
#imf_z42=emd(z1[42])
#imf_z43=emd(z1[43])
#imf_z44=emd(z1[44])
#imf_z45=emd(z1[45])
#imf_z46=emd(z1[46])
#imf_z47=emd(z1[47])
#imf_z48=emd(z1[48])
#imf_z49=emd(z1[49])
#imf_z50=emd(z1[50])
#imf_z51=emd(z1[51])
#imf_z52=emd(z1[52])
#imf_z53=emd(z1[53])
#imf_z54=emd(z1[54])
#imf_z55=emd(z1[55])
#imf_z56=emd(z1[56])
#imf_z57=emd(z1[57])
#imf_z58=emd(z1[58])
#imf_z59=emd(z1[59])
#imf_z60=emd(z1[60])
#imf_z61=emd(z1[61])
#imf_z62=emd(z1[62])
#imf_z63=emd(z1[63])
#imf_z64=emd(z1[64])
#imf_z65=emd(z1[65])
#imf_z66=emd(z1[66])
#imf_z67=emd(z1[67])
#imf_z68=emd(z1[68])
#imf_z69=emd(z1[69])
#imf_z70=emd(z1[70])
#imf_z71=emd(z1[71])
#imf_z72=emd(z1[72])
#imf_z73=emd(z1[73])
#imf_z74=emd(z1[74])
#imf_z75=emd(z1[75])
#imf_z76=emd(z1[76])
#imf_z77=emd(z1[77])
#imf_z78=emd(z1[78])
#imf_z79=emd(z1[79])
#imf_z80=emd(z1[80])
#imf_z81=emd(z1[81])
#imf_z82=emd(z1[82])
#imf_z83=emd(z1[83])
#imf_z84=emd(z1[84])
#imf_z85=emd(z1[85])
#imf_z86=emd(z1[86])
#imf_z87=emd(z1[87])
#imf_z88=emd(z1[88])
#imf_z89=emd(z1[89])
#imf_z90=emd(z1[90])
#imf_z91=emd(z1[91])
#imf_z92=emd(z1[92])
#imf_z93=emd(z1[93])
#imf_z94=emd(z1[94])
#imf_z95=emd(z1[95])
#imf_z96=emd(z1[96])
#imf_z97=emd(z1[97])
#imf_z98=emd(z1[98])
#imf_z99=emd(z1[99])
#imf_z100=emd(z1[100])
#imf_z101=emd(z1[101])
#imf_z102=emd(z1[102])
#imf_z103=emd(z1[103])
#imf_z104=emd(z1[104])
#imf_z105=emd(z1[105])
#imf_z106=emd(z1[106])
#imf_z107=emd(z1[107])
#imf_z108=emd(z1[108])
#imf_z109=emd(z1[109])
#imf_z110=emd(z1[110])
#imf_z111=emd(z1[111])
#imf_z112=emd(z1[112])
#imf_z113=emd(z1[113])
#imf_z114=emd(z1[114])
#imf_z115=emd(z1[115])
#imf_z116=emd(z1[116])
#imf_z117=emd(z1[117])
#imf_z118=emd(z1[118])
#imf_z119=emd(z1[119])
#imf_z120=emd(z1[120])
#imf_z121=emd(z1[121])
#imf_z122=emd(z1[122])
#imf_z123=emd(z1[123])
#imf_z124=emd(z1[124])
#imf_z125=emd(z1[125])
#imf_z126=emd(z1[126])
#imf_z127=emd(z1[127])
#imf_z128=emd(z1[128])
#imf_z129=emd(z1[129])
#imf_z130=emd(z1[130])
#imf_z131=emd(z1[131])
#imf_z132=emd(z1[132])
#imf_z133=emd(z1[133])
#imf_z134=emd(z1[134])
#imf_z135=emd(z1[135])
#imf_z136=emd(z1[136])
#imf_z137=emd(z1[137])
#imf_z138=emd(z1[138])
#imf_z139=emd(z1[139])
#imf_z140=emd(z1[140])
#imf_z141=emd(z1[141])
#imf_z142=emd(z1[142])
#imf_z143=emd(z1[143])
#imf_z144=emd(z1[144])
#imf_z145=emd(z1[145])
#imf_z146=emd(z1[146])
#imf_z147=emd(z1[147])
#imf_z148=emd(z1[148])
#imf_z149=emd(z1[149])
#imf_z150=emd(z1[150])
#imf_z151=emd(z1[151])
#imf_z152=emd(z1[152])
#imf_z153=emd(z1[153])
#imf_z154=emd(z1[154])
#imf_z155=emd(z1[155])
#imf_z156=emd(z1[156])
#imf_z157=emd(z1[157])
#imf_z158=emd(z1[158])
#imf_z159=emd(z1[159])
#imf_z160=emd(z1[160])
#imf_z161=emd(z1[161])
#imf_z162=emd(z1[162])
#imf_z163=emd(z1[163])
#imf_z164=emd(z1[164])
#imf_z165=emd(z1[165])
#imf_z166=emd(z1[166])
#imf_z167=emd(z1[167])
#imf_z168=emd(z1[168])
#imf_z169=emd(z1[169])
#imf_z170=emd(z1[170])
#imf_z171=emd(z1[171])
#imf_z172=emd(z1[172])
#imf_z173=emd(z1[173])
#imf_z174=emd(z1[174])
#imf_z175=emd(z1[175])
#imf_z176=emd(z1[176])
#imf_z177=emd(z1[177])
#imf_z178=emd(z1[178])
#imf_z179=emd(z1[179])
#imf_z180=emd(z1[180])
#imf_z181=emd(z1[181])
#imf_z182=emd(z1[182])
#imf_z183=emd(z1[183])
#imf_z184=emd(z1[184])
#imf_z185=emd(z1[185])
#imf_z186=emd(z1[186])
#imf_z187=emd(z1[187])
#imf_z188=emd(z1[188])
#imf_z189=emd(z1[189])
#imf_z190=emd(z1[190])
#imf_z191=emd(z1[191])
#imf_z192=emd(z1[192])
#imf_z193=emd(z1[193])
#imf_z194=emd(z1[194])
#imf_z195=emd(z1[195])
#imf_z196=emd(z1[196])
#imf_z197=emd(z1[197])
#imf_z198=emd(z1[198])
#imf_z199=emd(z1[199])
#imf_z200=emd(z1[200])
#imf_z201=emd(z1[201])
#imf_z202=emd(z1[202])
#imf_z203=emd(z1[203])
#imf_z204=emd(z1[204])
#imf_z205=emd(z1[205])
#imf_z206=emd(z1[206])
#imf_z207=emd(z1[207])
#imf_z208=emd(z1[208])
#imf_z209=emd(z1[209])
#imf_z210=emd(z1[210])
#imf_z211=emd(z1[211])
#imf_z212=emd(z1[212])
#imf_z213=emd(z1[213])
#imf_z214=emd(z1[214])
#imf_z215=emd(z1[215])
#imf_z216=emd(z1[216])
#imf_z217=emd(z1[217])
#imf_z218=emd(z1[218])
#imf_z219=emd(z1[219])
#imf_z220=emd(z1[220])
#imf_z221=emd(z1[221])
#imf_z222=emd(z1[222])
#imf_z223=emd(z1[223])
#imf_z224=emd(z1[224])
#imf_z225=emd(z1[225])
#imf_z226=emd(z1[226])
#imf_z227=emd(z1[227])
#imf_z228=emd(z1[228])
#imf_z229=emd(z1[229])
#imf_z230=emd(z1[230])
#imf_z231=emd(z1[231])
#imf_z232=emd(z1[232])
#imf_z233=emd(z1[233])
#imf_z234=emd(z1[234])
#imf_z235=emd(z1[235])
#imf_z236=emd(z1[236])
#imf_z237=emd(z1[237])
#imf_z238=emd(z1[238])
#imf_z239=emd(z1[239])
#imf_z240=emd(z1[240])
#imf_z241=emd(z1[241])
#imf_z242=emd(z1[242])
#imf_z243=emd(z1[243])
#imf_z244=emd(z1[244])
#imf_z245=emd(z1[245])
#imf_z246=emd(z1[246])
#imf_z247=emd(z1[247])
#imf_z248=emd(z1[248])
#imf_z249=emd(z1[249])
#imf_z250=emd(z1[250])
#imf_z251=emd(z1[251])
#imf_z252=emd(z1[252])
#imf_z253=emd(z1[253])
#imf_z254=emd(z1[254])
#imf_z255=emd(z1[255])
#imf_z256=emd(z1[256])
#imf_z257=emd(z1[257])
#imf_z258=emd(z1[258])
#imf_z259=emd(z1[259])
#imf_z260=emd(z1[260])
#imf_z261=emd(z1[261])
#imf_z262=emd(z1[262])
#imf_z263=emd(z1[263])
#imf_z264=emd(z1[264])
#imf_z265=emd(z1[265])
#imf_z266=emd(z1[266])
#imf_z267=emd(z1[267])
#imf_z268=emd(z1[268])
#imf_z269=emd(z1[269])
#imf_z270=emd(z1[270])
#imf_z271=emd(z1[271])
#imf_z272=emd(z1[272])
#imf_z273=emd(z1[273])
#imf_z274=emd(z1[274])
#imf_z275=emd(z1[275])
#imf_z276=emd(z1[276])
#imf_z277=emd(z1[277])
#imf_z278=emd(z1[278])
#imf_z279=emd(z1[279])
#imf_z280=emd(z1[280])
#imf_z281=emd(z1[281])
#imf_z282=emd(z1[282])
#imf_z283=emd(z1[283])
#imf_z284=emd(z1[284])
#imf_z285=emd(z1[285])
#imf_z286=emd(z1[286])
#imf_z287=emd(z1[287])
#imf_z288=emd(z1[288])
#imf_z289=emd(z1[289])
#imf_z290=emd(z1[290])
#imf_z291=emd(z1[291])
#imf_z292=emd(z1[292])
#imf_z293=emd(z1[293])
#imf_z294=emd(z1[294])
#imf_z295=emd(z1[295])
#imf_z296=emd(z1[296])
#imf_z297=emd(z1[297])
#imf_z298=emd(z1[298])
#imf_z299=emd(z1[299])
#imf_z300=emd(z1[300])
#imf_z301=emd(z1[301])
#imf_z302=emd(z1[302])
#imf_z303=emd(z1[303])
#imf_z304=emd(z1[304])
#imf_z305=emd(z1[305])
#imf_z306=emd(z1[306])
#imf_z307=emd(z1[307])
#imf_z308=emd(z1[308])
#imf_z309=emd(z1[309])
#imf_z310=emd(z1[310])
#imf_z311=emd(z1[311])
#imf_z312=emd(z1[312])
#imf_z313=emd(z1[313])
#imf_z314=emd(z1[314])
#print('imf_z_finish')
#data00=np.vstack((imf_x0[0],imf_y0[0],imf_z0[0]))
#data01=np.vstack((imf_x1[0],imf_y1[0],imf_z1[0]))
#data02=np.vstack((imf_x2[0],imf_y2[0],imf_z2[0]))
#data03=np.vstack((imf_x3[0],imf_y3[0],imf_z3[0]))
#data04=np.vstack((imf_x4[0],imf_y4[0],imf_z4[0]))
#data05=np.vstack((imf_x5[0],imf_y5[0],imf_z5[0]))
#data06=np.vstack((imf_x6[0],imf_y6[0],imf_z6[0]))
#data07=np.vstack((imf_x7[0],imf_y7[0],imf_z7[0]))
#data08=np.vstack((imf_x8[0],imf_y8[0],imf_z8[0]))
#data09=np.vstack((imf_x9[0],imf_y9[0],imf_z9[0]))
#data010=np.vstack((imf_x10[0],imf_y10[0],imf_z10[0]))
#data011=np.vstack((imf_x11[0],imf_y11[0],imf_z11[0]))
#data012=np.vstack((imf_x12[0],imf_y12[0],imf_z12[0]))
#data013=np.vstack((imf_x13[0],imf_y13[0],imf_z13[0]))
#data014=np.vstack((imf_x14[0],imf_y14[0],imf_z14[0]))
#data015=np.vstack((imf_x15[0],imf_y15[0],imf_z15[0]))
#data016=np.vstack((imf_x16[0],imf_y16[0],imf_z16[0]))
#data017=np.vstack((imf_x17[0],imf_y17[0],imf_z17[0]))
#data018=np.vstack((imf_x18[0],imf_y18[0],imf_z18[0]))
#data019=np.vstack((imf_x19[0],imf_y19[0],imf_z19[0]))
#data020=np.vstack((imf_x20[0],imf_y20[0],imf_z20[0]))
#data021=np.vstack((imf_x21[0],imf_y21[0],imf_z21[0]))
#data022=np.vstack((imf_x22[0],imf_y22[0],imf_z22[0]))
#data023=np.vstack((imf_x23[0],imf_y23[0],imf_z23[0]))
#data024=np.vstack((imf_x24[0],imf_y24[0],imf_z24[0]))
#data025=np.vstack((imf_x25[0],imf_y25[0],imf_z25[0]))
#data026=np.vstack((imf_x26[0],imf_y26[0],imf_z26[0]))
#data027=np.vstack((imf_x27[0],imf_y27[0],imf_z27[0]))
#data028=np.vstack((imf_x28[0],imf_y28[0],imf_z28[0]))
#data029=np.vstack((imf_x29[0],imf_y29[0],imf_z29[0]))
#data030=np.vstack((imf_x30[0],imf_y30[0],imf_z30[0]))
#data031=np.vstack((imf_x31[0],imf_y31[0],imf_z31[0]))
#data032=np.vstack((imf_x32[0],imf_y32[0],imf_z32[0]))
#data033=np.vstack((imf_x33[0],imf_y33[0],imf_z33[0]))
#data034=np.vstack((imf_x34[0],imf_y34[0],imf_z34[0]))
#data035=np.vstack((imf_x35[0],imf_y35[0],imf_z35[0]))
#data036=np.vstack((imf_x36[0],imf_y36[0],imf_z36[0]))
#data037=np.vstack((imf_x37[0],imf_y37[0],imf_z37[0]))
#data038=np.vstack((imf_x38[0],imf_y38[0],imf_z38[0]))
#data039=np.vstack((imf_x39[0],imf_y39[0],imf_z39[0]))
#data040=np.vstack((imf_x40[0],imf_y40[0],imf_z40[0]))
#data041=np.vstack((imf_x41[0],imf_y41[0],imf_z41[0]))
#data042=np.vstack((imf_x42[0],imf_y42[0],imf_z42[0]))
#data043=np.vstack((imf_x43[0],imf_y43[0],imf_z43[0]))
#data044=np.vstack((imf_x44[0],imf_y44[0],imf_z44[0]))
#data045=np.vstack((imf_x45[0],imf_y45[0],imf_z45[0]))
#data046=np.vstack((imf_x46[0],imf_y46[0],imf_z46[0]))
#data047=np.vstack((imf_x47[0],imf_y47[0],imf_z47[0]))
#data048=np.vstack((imf_x48[0],imf_y48[0],imf_z48[0]))
#data049=np.vstack((imf_x49[0],imf_y49[0],imf_z49[0]))
#data050=np.vstack((imf_x50[0],imf_y50[0],imf_z50[0]))
#data051=np.vstack((imf_x51[0],imf_y51[0],imf_z51[0]))
#data052=np.vstack((imf_x52[0],imf_y52[0],imf_z52[0]))
#data053=np.vstack((imf_x53[0],imf_y53[0],imf_z53[0]))
#data054=np.vstack((imf_x54[0],imf_y54[0],imf_z54[0]))
#data055=np.vstack((imf_x55[0],imf_y55[0],imf_z55[0]))
#data056=np.vstack((imf_x56[0],imf_y56[0],imf_z56[0]))
#data057=np.vstack((imf_x57[0],imf_y57[0],imf_z57[0]))
#data058=np.vstack((imf_x58[0],imf_y58[0],imf_z58[0]))
#data059=np.vstack((imf_x59[0],imf_y59[0],imf_z59[0]))
#data060=np.vstack((imf_x60[0],imf_y60[0],imf_z60[0]))
#data061=np.vstack((imf_x61[0],imf_y61[0],imf_z61[0]))
#data062=np.vstack((imf_x62[0],imf_y62[0],imf_z62[0]))
#data063=np.vstack((imf_x63[0],imf_y63[0],imf_z63[0]))
#data064=np.vstack((imf_x64[0],imf_y64[0],imf_z64[0]))
#data065=np.vstack((imf_x65[0],imf_y65[0],imf_z65[0]))
#data066=np.vstack((imf_x66[0],imf_y66[0],imf_z66[0]))
#data067=np.vstack((imf_x67[0],imf_y67[0],imf_z67[0]))
#data068=np.vstack((imf_x68[0],imf_y68[0],imf_z68[0]))
#data069=np.vstack((imf_x69[0],imf_y69[0],imf_z69[0]))
#data070=np.vstack((imf_x70[0],imf_y70[0],imf_z70[0]))
#data071=np.vstack((imf_x71[0],imf_y71[0],imf_z71[0]))
#data072=np.vstack((imf_x72[0],imf_y72[0],imf_z72[0]))
#data073=np.vstack((imf_x73[0],imf_y73[0],imf_z73[0]))
#data074=np.vstack((imf_x74[0],imf_y74[0],imf_z74[0]))
#data075=np.vstack((imf_x75[0],imf_y75[0],imf_z75[0]))
#data076=np.vstack((imf_x76[0],imf_y76[0],imf_z76[0]))
#data077=np.vstack((imf_x77[0],imf_y77[0],imf_z77[0]))
#data078=np.vstack((imf_x78[0],imf_y78[0],imf_z78[0]))
#data079=np.vstack((imf_x79[0],imf_y79[0],imf_z79[0]))
#data080=np.vstack((imf_x80[0],imf_y80[0],imf_z80[0]))
#data081=np.vstack((imf_x81[0],imf_y81[0],imf_z81[0]))
#data082=np.vstack((imf_x82[0],imf_y82[0],imf_z82[0]))
#data083=np.vstack((imf_x83[0],imf_y83[0],imf_z83[0]))
#data084=np.vstack((imf_x84[0],imf_y84[0],imf_z84[0]))
#data085=np.vstack((imf_x85[0],imf_y85[0],imf_z85[0]))
#data086=np.vstack((imf_x86[0],imf_y86[0],imf_z86[0]))
#data087=np.vstack((imf_x87[0],imf_y87[0],imf_z87[0]))
#data088=np.vstack((imf_x88[0],imf_y88[0],imf_z88[0]))
#data089=np.vstack((imf_x89[0],imf_y89[0],imf_z89[0]))
#data090=np.vstack((imf_x90[0],imf_y90[0],imf_z90[0]))
#data091=np.vstack((imf_x91[0],imf_y91[0],imf_z91[0]))
#data092=np.vstack((imf_x92[0],imf_y92[0],imf_z92[0]))
#data093=np.vstack((imf_x93[0],imf_y93[0],imf_z93[0]))
#data094=np.vstack((imf_x94[0],imf_y94[0],imf_z94[0]))
#data095=np.vstack((imf_x95[0],imf_y95[0],imf_z95[0]))
#data096=np.vstack((imf_x96[0],imf_y96[0],imf_z96[0]))
#data097=np.vstack((imf_x97[0],imf_y97[0],imf_z97[0]))
#data098=np.vstack((imf_x98[0],imf_y98[0],imf_z98[0]))
#data099=np.vstack((imf_x99[0],imf_y99[0],imf_z99[0]))
#data0100=np.vstack((imf_x100[0],imf_y100[0],imf_z100[0]))
#data0101=np.vstack((imf_x101[0],imf_y101[0],imf_z101[0]))
#data0102=np.vstack((imf_x102[0],imf_y102[0],imf_z102[0]))
#data0103=np.vstack((imf_x103[0],imf_y103[0],imf_z103[0]))
#data0104=np.vstack((imf_x104[0],imf_y104[0],imf_z104[0]))
#data0105=np.vstack((imf_x105[0],imf_y105[0],imf_z105[0]))
#data0106=np.vstack((imf_x106[0],imf_y106[0],imf_z106[0]))
#data0107=np.vstack((imf_x107[0],imf_y107[0],imf_z107[0]))
#data0108=np.vstack((imf_x108[0],imf_y108[0],imf_z108[0]))
#data0109=np.vstack((imf_x109[0],imf_y109[0],imf_z109[0]))
#data0110=np.vstack((imf_x110[0],imf_y110[0],imf_z110[0]))
#data0111=np.vstack((imf_x111[0],imf_y111[0],imf_z111[0]))
#data0112=np.vstack((imf_x112[0],imf_y112[0],imf_z112[0]))
#data0113=np.vstack((imf_x113[0],imf_y113[0],imf_z113[0]))
#data0114=np.vstack((imf_x114[0],imf_y114[0],imf_z114[0]))
#data0115=np.vstack((imf_x115[0],imf_y115[0],imf_z115[0]))
#data0116=np.vstack((imf_x116[0],imf_y116[0],imf_z116[0]))
#data0117=np.vstack((imf_x117[0],imf_y117[0],imf_z117[0]))
#data0118=np.vstack((imf_x118[0],imf_y118[0],imf_z118[0]))
#data0119=np.vstack((imf_x119[0],imf_y119[0],imf_z119[0]))
#data0120=np.vstack((imf_x120[0],imf_y120[0],imf_z120[0]))
#data0121=np.vstack((imf_x121[0],imf_y121[0],imf_z121[0]))
#data0122=np.vstack((imf_x122[0],imf_y122[0],imf_z122[0]))
#data0123=np.vstack((imf_x123[0],imf_y123[0],imf_z123[0]))
#data0124=np.vstack((imf_x124[0],imf_y124[0],imf_z124[0]))
#data0125=np.vstack((imf_x125[0],imf_y125[0],imf_z125[0]))
#data0126=np.vstack((imf_x126[0],imf_y126[0],imf_z126[0]))
#data0127=np.vstack((imf_x127[0],imf_y127[0],imf_z127[0]))
#data0128=np.vstack((imf_x128[0],imf_y128[0],imf_z128[0]))
#data0129=np.vstack((imf_x129[0],imf_y129[0],imf_z129[0]))
#data0130=np.vstack((imf_x130[0],imf_y130[0],imf_z130[0]))
#data0131=np.vstack((imf_x131[0],imf_y131[0],imf_z131[0]))
#data0132=np.vstack((imf_x132[0],imf_y132[0],imf_z132[0]))
#data0133=np.vstack((imf_x133[0],imf_y133[0],imf_z133[0]))
#data0134=np.vstack((imf_x134[0],imf_y134[0],imf_z134[0]))
#data0135=np.vstack((imf_x135[0],imf_y135[0],imf_z135[0]))
#data0136=np.vstack((imf_x136[0],imf_y136[0],imf_z136[0]))
#data0137=np.vstack((imf_x137[0],imf_y137[0],imf_z137[0]))
#data0138=np.vstack((imf_x138[0],imf_y138[0],imf_z138[0]))
#data0139=np.vstack((imf_x139[0],imf_y139[0],imf_z139[0]))
#data0140=np.vstack((imf_x140[0],imf_y140[0],imf_z140[0]))
#data0141=np.vstack((imf_x141[0],imf_y141[0],imf_z141[0]))
#data0142=np.vstack((imf_x142[0],imf_y142[0],imf_z142[0]))
#data0143=np.vstack((imf_x143[0],imf_y143[0],imf_z143[0]))
#data0144=np.vstack((imf_x144[0],imf_y144[0],imf_z144[0]))
#data0145=np.vstack((imf_x145[0],imf_y145[0],imf_z145[0]))
#data0146=np.vstack((imf_x146[0],imf_y146[0],imf_z146[0]))
#data0147=np.vstack((imf_x147[0],imf_y147[0],imf_z147[0]))
#data0148=np.vstack((imf_x148[0],imf_y148[0],imf_z148[0]))
#data0149=np.vstack((imf_x149[0],imf_y149[0],imf_z149[0]))
#data0150=np.vstack((imf_x150[0],imf_y150[0],imf_z150[0]))
#data0151=np.vstack((imf_x151[0],imf_y151[0],imf_z151[0]))
#data0152=np.vstack((imf_x152[0],imf_y152[0],imf_z152[0]))
#data0153=np.vstack((imf_x153[0],imf_y153[0],imf_z153[0]))
#data0154=np.vstack((imf_x154[0],imf_y154[0],imf_z154[0]))
#data0155=np.vstack((imf_x155[0],imf_y155[0],imf_z155[0]))
#data0156=np.vstack((imf_x156[0],imf_y156[0],imf_z156[0]))
#data0157=np.vstack((imf_x157[0],imf_y157[0],imf_z157[0]))
#data0158=np.vstack((imf_x158[0],imf_y158[0],imf_z158[0]))
#data0159=np.vstack((imf_x159[0],imf_y159[0],imf_z159[0]))
#data0160=np.vstack((imf_x160[0],imf_y160[0],imf_z160[0]))
#data0161=np.vstack((imf_x161[0],imf_y161[0],imf_z161[0]))
#data0162=np.vstack((imf_x162[0],imf_y162[0],imf_z162[0]))
#data0163=np.vstack((imf_x163[0],imf_y163[0],imf_z163[0]))
#data0164=np.vstack((imf_x164[0],imf_y164[0],imf_z164[0]))
#data0165=np.vstack((imf_x165[0],imf_y165[0],imf_z165[0]))
#data0166=np.vstack((imf_x166[0],imf_y166[0],imf_z166[0]))
#data0167=np.vstack((imf_x167[0],imf_y167[0],imf_z167[0]))
#data0168=np.vstack((imf_x168[0],imf_y168[0],imf_z168[0]))
#data0169=np.vstack((imf_x169[0],imf_y169[0],imf_z169[0]))
#data0170=np.vstack((imf_x170[0],imf_y170[0],imf_z170[0]))
#data0171=np.vstack((imf_x171[0],imf_y171[0],imf_z171[0]))
#data0172=np.vstack((imf_x172[0],imf_y172[0],imf_z172[0]))
#data0173=np.vstack((imf_x173[0],imf_y173[0],imf_z173[0]))
#data0174=np.vstack((imf_x174[0],imf_y174[0],imf_z174[0]))
#data0175=np.vstack((imf_x175[0],imf_y175[0],imf_z175[0]))
#data0176=np.vstack((imf_x176[0],imf_y176[0],imf_z176[0]))
#data0177=np.vstack((imf_x177[0],imf_y177[0],imf_z177[0]))
#data0178=np.vstack((imf_x178[0],imf_y178[0],imf_z178[0]))
#data0179=np.vstack((imf_x179[0],imf_y179[0],imf_z179[0]))
#data0180=np.vstack((imf_x180[0],imf_y180[0],imf_z180[0]))
#data0181=np.vstack((imf_x181[0],imf_y181[0],imf_z181[0]))
#data0182=np.vstack((imf_x182[0],imf_y182[0],imf_z182[0]))
#data0183=np.vstack((imf_x183[0],imf_y183[0],imf_z183[0]))
#data0184=np.vstack((imf_x184[0],imf_y184[0],imf_z184[0]))
#data0185=np.vstack((imf_x185[0],imf_y185[0],imf_z185[0]))
#data0186=np.vstack((imf_x186[0],imf_y186[0],imf_z186[0]))
#data0187=np.vstack((imf_x187[0],imf_y187[0],imf_z187[0]))
#data0188=np.vstack((imf_x188[0],imf_y188[0],imf_z188[0]))
#data0189=np.vstack((imf_x189[0],imf_y189[0],imf_z189[0]))
#data0190=np.vstack((imf_x190[0],imf_y190[0],imf_z190[0]))
#data0191=np.vstack((imf_x191[0],imf_y191[0],imf_z191[0]))
#data0192=np.vstack((imf_x192[0],imf_y192[0],imf_z192[0]))
#data0193=np.vstack((imf_x193[0],imf_y193[0],imf_z193[0]))
#data0194=np.vstack((imf_x194[0],imf_y194[0],imf_z194[0]))
#data0195=np.vstack((imf_x195[0],imf_y195[0],imf_z195[0]))
#data0196=np.vstack((imf_x196[0],imf_y196[0],imf_z196[0]))
#data0197=np.vstack((imf_x197[0],imf_y197[0],imf_z197[0]))
#data0198=np.vstack((imf_x198[0],imf_y198[0],imf_z198[0]))
#data0199=np.vstack((imf_x199[0],imf_y199[0],imf_z199[0]))
#data0200=np.vstack((imf_x200[0],imf_y200[0],imf_z200[0]))
#data0201=np.vstack((imf_x201[0],imf_y201[0],imf_z201[0]))
#data0202=np.vstack((imf_x202[0],imf_y202[0],imf_z202[0]))
#data0203=np.vstack((imf_x203[0],imf_y203[0],imf_z203[0]))
#data0204=np.vstack((imf_x204[0],imf_y204[0],imf_z204[0]))
#data0205=np.vstack((imf_x205[0],imf_y205[0],imf_z205[0]))
#data0206=np.vstack((imf_x206[0],imf_y206[0],imf_z206[0]))
#data0207=np.vstack((imf_x207[0],imf_y207[0],imf_z207[0]))
#data0208=np.vstack((imf_x208[0],imf_y208[0],imf_z208[0]))
#data0209=np.vstack((imf_x209[0],imf_y209[0],imf_z209[0]))
#data0210=np.vstack((imf_x210[0],imf_y210[0],imf_z210[0]))
#data0211=np.vstack((imf_x211[0],imf_y211[0],imf_z211[0]))
#data0212=np.vstack((imf_x212[0],imf_y212[0],imf_z212[0]))
#data0213=np.vstack((imf_x213[0],imf_y213[0],imf_z213[0]))
#data0214=np.vstack((imf_x214[0],imf_y214[0],imf_z214[0]))
#data0215=np.vstack((imf_x215[0],imf_y215[0],imf_z215[0]))
#data0216=np.vstack((imf_x216[0],imf_y216[0],imf_z216[0]))
#data0217=np.vstack((imf_x217[0],imf_y217[0],imf_z217[0]))
#data0218=np.vstack((imf_x218[0],imf_y218[0],imf_z218[0]))
#data0219=np.vstack((imf_x219[0],imf_y219[0],imf_z219[0]))
#data0220=np.vstack((imf_x220[0],imf_y220[0],imf_z220[0]))
#data0221=np.vstack((imf_x221[0],imf_y221[0],imf_z221[0]))
#data0222=np.vstack((imf_x222[0],imf_y222[0],imf_z222[0]))
#data0223=np.vstack((imf_x223[0],imf_y223[0],imf_z223[0]))
#data0224=np.vstack((imf_x224[0],imf_y224[0],imf_z224[0]))
#data0225=np.vstack((imf_x225[0],imf_y225[0],imf_z225[0]))
#data0226=np.vstack((imf_x226[0],imf_y226[0],imf_z226[0]))
#data0227=np.vstack((imf_x227[0],imf_y227[0],imf_z227[0]))
#data0228=np.vstack((imf_x228[0],imf_y228[0],imf_z228[0]))
#data0229=np.vstack((imf_x229[0],imf_y229[0],imf_z229[0]))
#data0230=np.vstack((imf_x230[0],imf_y230[0],imf_z230[0]))
#data0231=np.vstack((imf_x231[0],imf_y231[0],imf_z231[0]))
#data0232=np.vstack((imf_x232[0],imf_y232[0],imf_z232[0]))
#data0233=np.vstack((imf_x233[0],imf_y233[0],imf_z233[0]))
#data0234=np.vstack((imf_x234[0],imf_y234[0],imf_z234[0]))
#data0235=np.vstack((imf_x235[0],imf_y235[0],imf_z235[0]))
#data0236=np.vstack((imf_x236[0],imf_y236[0],imf_z236[0]))
#data0237=np.vstack((imf_x237[0],imf_y237[0],imf_z237[0]))
#data0238=np.vstack((imf_x238[0],imf_y238[0],imf_z238[0]))
#data0239=np.vstack((imf_x239[0],imf_y239[0],imf_z239[0]))
#data0240=np.vstack((imf_x240[0],imf_y240[0],imf_z240[0]))
#data0241=np.vstack((imf_x241[0],imf_y241[0],imf_z241[0]))
#data0242=np.vstack((imf_x242[0],imf_y242[0],imf_z242[0]))
#data0243=np.vstack((imf_x243[0],imf_y243[0],imf_z243[0]))
#data0244=np.vstack((imf_x244[0],imf_y244[0],imf_z244[0]))
#data0245=np.vstack((imf_x245[0],imf_y245[0],imf_z245[0]))
#data0246=np.vstack((imf_x246[0],imf_y246[0],imf_z246[0]))
#data0247=np.vstack((imf_x247[0],imf_y247[0],imf_z247[0]))
#data0248=np.vstack((imf_x248[0],imf_y248[0],imf_z248[0]))
#data0249=np.vstack((imf_x249[0],imf_y249[0],imf_z249[0]))
#data0250=np.vstack((imf_x250[0],imf_y250[0],imf_z250[0]))
#data0251=np.vstack((imf_x251[0],imf_y251[0],imf_z251[0]))
#data0252=np.vstack((imf_x252[0],imf_y252[0],imf_z252[0]))
#data0253=np.vstack((imf_x253[0],imf_y253[0],imf_z253[0]))
#data0254=np.vstack((imf_x254[0],imf_y254[0],imf_z254[0]))
#data0255=np.vstack((imf_x255[0],imf_y255[0],imf_z255[0]))
#data0256=np.vstack((imf_x256[0],imf_y256[0],imf_z256[0]))
#data0257=np.vstack((imf_x257[0],imf_y257[0],imf_z257[0]))
#data0258=np.vstack((imf_x258[0],imf_y258[0],imf_z258[0]))
#data0259=np.vstack((imf_x259[0],imf_y259[0],imf_z259[0]))
#data0260=np.vstack((imf_x260[0],imf_y260[0],imf_z260[0]))
#data0261=np.vstack((imf_x261[0],imf_y261[0],imf_z261[0]))
#data0262=np.vstack((imf_x262[0],imf_y262[0],imf_z262[0]))
#data0263=np.vstack((imf_x263[0],imf_y263[0],imf_z263[0]))
#data0264=np.vstack((imf_x264[0],imf_y264[0],imf_z264[0]))
#data0265=np.vstack((imf_x265[0],imf_y265[0],imf_z265[0]))
#data0266=np.vstack((imf_x266[0],imf_y266[0],imf_z266[0]))
#data0267=np.vstack((imf_x267[0],imf_y267[0],imf_z267[0]))
#data0268=np.vstack((imf_x268[0],imf_y268[0],imf_z268[0]))
#data0269=np.vstack((imf_x269[0],imf_y269[0],imf_z269[0]))
#data0270=np.vstack((imf_x270[0],imf_y270[0],imf_z270[0]))
#data0271=np.vstack((imf_x271[0],imf_y271[0],imf_z271[0]))
#data0272=np.vstack((imf_x272[0],imf_y272[0],imf_z272[0]))
#data0273=np.vstack((imf_x273[0],imf_y273[0],imf_z273[0]))
#data0274=np.vstack((imf_x274[0],imf_y274[0],imf_z274[0]))
#data0275=np.vstack((imf_x275[0],imf_y275[0],imf_z275[0]))
#data0276=np.vstack((imf_x276[0],imf_y276[0],imf_z276[0]))
#data0277=np.vstack((imf_x277[0],imf_y277[0],imf_z277[0]))
#data0278=np.vstack((imf_x278[0],imf_y278[0],imf_z278[0]))
#data0279=np.vstack((imf_x279[0],imf_y279[0],imf_z279[0]))
#data0280=np.vstack((imf_x280[0],imf_y280[0],imf_z280[0]))
#data0281=np.vstack((imf_x281[0],imf_y281[0],imf_z281[0]))
#data0282=np.vstack((imf_x282[0],imf_y282[0],imf_z282[0]))
#data0283=np.vstack((imf_x283[0],imf_y283[0],imf_z283[0]))
#data0284=np.vstack((imf_x284[0],imf_y284[0],imf_z284[0]))
#data0285=np.vstack((imf_x285[0],imf_y285[0],imf_z285[0]))
#data0286=np.vstack((imf_x286[0],imf_y286[0],imf_z286[0]))
#data0287=np.vstack((imf_x287[0],imf_y287[0],imf_z287[0]))
#data0288=np.vstack((imf_x288[0],imf_y288[0],imf_z288[0]))
#data0289=np.vstack((imf_x289[0],imf_y289[0],imf_z289[0]))
#data0290=np.vstack((imf_x290[0],imf_y290[0],imf_z290[0]))
#data0291=np.vstack((imf_x291[0],imf_y291[0],imf_z291[0]))
#data0292=np.vstack((imf_x292[0],imf_y292[0],imf_z292[0]))
#data0293=np.vstack((imf_x293[0],imf_y293[0],imf_z293[0]))
#data0294=np.vstack((imf_x294[0],imf_y294[0],imf_z294[0]))
#data0295=np.vstack((imf_x295[0],imf_y295[0],imf_z295[0]))
#data0296=np.vstack((imf_x296[0],imf_y296[0],imf_z296[0]))
#data0297=np.vstack((imf_x297[0],imf_y297[0],imf_z297[0]))
#data0298=np.vstack((imf_x298[0],imf_y298[0],imf_z298[0]))
#data0299=np.vstack((imf_x299[0],imf_y299[0],imf_z299[0]))
#data0300=np.vstack((imf_x300[0],imf_y300[0],imf_z300[0]))
#data0301=np.vstack((imf_x301[0],imf_y301[0],imf_z301[0]))
#data0302=np.vstack((imf_x302[0],imf_y302[0],imf_z302[0]))
#data0303=np.vstack((imf_x303[0],imf_y303[0],imf_z303[0]))
#data0304=np.vstack((imf_x304[0],imf_y304[0],imf_z304[0]))
#data0305=np.vstack((imf_x305[0],imf_y305[0],imf_z305[0]))
#data0306=np.vstack((imf_x306[0],imf_y306[0],imf_z306[0]))
#data0307=np.vstack((imf_x307[0],imf_y307[0],imf_z307[0]))
#data0308=np.vstack((imf_x308[0],imf_y308[0],imf_z308[0]))
#data0309=np.vstack((imf_x309[0],imf_y309[0],imf_z309[0]))
#data0310=np.vstack((imf_x310[0],imf_y310[0],imf_z310[0]))
#data0311=np.vstack((imf_x311[0],imf_y311[0],imf_z311[0]))
#data0312=np.vstack((imf_x312[0],imf_y312[0],imf_z312[0]))
#data0313=np.vstack((imf_x313[0],imf_y313[0],imf_z313[0]))
#data0314=np.vstack((imf_x314[0],imf_y314[0],imf_z314[0]))
#print('imf_vstack_finish')
#data=hstack((data00,data01,data02,data03,data04,data05,data06,data07,data08,data09,data010,
#             data011,data012,data013,data014,data015,data016,data017,data018,data019,data020,
#             data021,data022,data023,data024,data025,data026,data027,data028,data029,data030,
#             data031,data032,data033,data034,data035,data036,data037,data038,data039,data040,
#             data041,data042,data043,data044,data045,data046,data047,data048,data049,data050,
#             data051,data052,data053,data054,data055,data056,data057,data058,data059,data060,
#             data061,data062,data063,data064,data065,data066,data067,data068,data069,data070,
#             data071,data072,data073,data074,data075,data076,data077,data078,data079,data080,
#             data081,data082,data083,data084,data085,data086,data087,data088,data089,data090,
#             data091,data092,data093,data094,data095,data096,data097,data098,data099,data0100,
#             data0101,data0102,data0103,data0104,data0105,data0106,data0107,data0108,data0109,
#             data0110,data0111,data0112,data0113,data0114,data0115,data0116,data0117,data0118,
#             data0119,data0120,data0121,data0122,data0123,data0124,data0125,data0126,data0127,
#             data0128,data0129,data0130,data0131,data0132,data0133,data0134,data0135,data0136,
#             data0137,data0138,data0139,data0140,data0141,data0142,data0143,data0144,data0145,
#             data0146,data0147,data0148,data0149,data0150,data0151,data0152,data0153,data0154,
#             data0155,data0156,data0157,data0158,data0159,data0160,data0161,data0162,data0163,
#             data0164,data0165,data0166,data0167,data0168,data0169,data0170,data0171,data0172,
#             data0173,data0174,data0175,data0176,data0177,data0178,data0179,data0180,data0181,
#             data0182,data0183,data0184,data0185,data0186,data0187,data0188,data0189,data0190,data0191,
#             data0192,data0193,data0194,data0195,data0196,data0197,data0198,data0199,data0200,data0201,
#             data0202,data0203,data0204,data0205,data0206,data0207,data0208,data0209,data0210,data0211,
#             data0212,data0213,data0214,data0215,data0216,data0217,data0218,data0219,data0220,data0221,
#             data0222,data0223,data0224,data0225,data0226,data0227,data0228,data0229,data0230,data0231,
#             data0232,data0233,data0234,data0235,data0236,data0237,data0238,data0239,data0240,data0241,
#             data0242,data0243,data0244,data0245,data0246,data0247,data0248,data0249,data0250,data0251,
#             data0252,data0253,data0254,data0255,data0256,data0257,data0258,data0259,data0260,data0261,
#             data0262,data0263,data0264,data0265,data0266,data0267,data0268,data0269,data0270,data0271,
#             data0272,data0273,data0274,data0275,data0276,data0277,data0278,data0279,data0280,data0281,
#             data0282,data0283,data0284,data0285,data0286,data0287,data0288,data0289,data0290,data0291,
#             data0292,data0293,data0294,data0295,data0296,data0297,data0298,data0299,data0300,data0301,
#             data0302,data0303,data0304,data0305,data0306,data0307,data0308,data0309,data0310,data0311,
#             data0312,data0313,data0314))

# =============================================================================
# print('imf_hstack_finish')
# print(len('imf_x0',imf_x0))
# print('data00.shape',data00.shape)
# print('data.shape',data.shape)
# print('data.shape',data.shape)
# print('data.shape',data.shape)
# 
# plt.figure(figsize=(16,4))
# plt.plot(t,x1[0])
# # plt.ylabel("Volt")
# plt.legend()
# plt.show()
# 
# plt.figure(figsize=(16,4))
# plt.plot(t,imf_x0[0])
# plt.ylabel("C2")
# plt.legend()
# plt.show()
# =============================================================================




'''
b,a = signal.butter(1,0.004,'lowpass')
filtedData = signal.filtfilt(b,a,x1[0])

plt.figure(figsize=(16,4))
plt.plot(t,filtedData)
# plt.ylabel("Volt")
plt.legend()
plt.show()
'''




'''
在使用Python进行信号处理过程中，利用 scipy.signal.filtfilt()可以快速帮助实现信号的滤波。

1.函数的介绍

(1).滤波函数

scipy.signal.filtfilt(b, a, x, axis=-1, padtype='odd', padlen=None, method='pad', irlen=None)

输入参数：

b: 滤波器的分子系数向量

a: 滤波器的分母系数向量

x: 要过滤的数据数组。（array型）

axis: 指定要过滤的数据数组x的轴

padtype: 必须是“奇数”、“偶数”、“常数”或“无”。这决定了用于过滤器应用的填充信号的扩展类型。{‘odd’, ‘even’, ‘constant’, None}

padlen：在应用滤波器之前在轴两端延伸X的元素数目。此值必须小于要滤波元素个数- 1。（int型或None）

method：确定处理信号边缘的方法。当method为“pad”时，填充信号；填充类型padtype和padlen决定，irlen被忽略。当method为“gust”时，使用古斯塔夫森方法，而忽略padtype和padlen。{“pad” ，“gust”}

irlen：当method为“gust”时，irlen指定滤波器的脉冲响应的长度。如果irlen是None，则脉冲响应的任何部分都被忽略。对于长信号，指定irlen可以显著改善滤波器的性能。（int型或None）

输出参数：

y:滤波后的数据数组

（2）.滤波器构造函数(仅介绍Butterworth滤波器)

scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba')

输入参数：

N:滤波器的阶数

Wn：归一化截止频率。计算公式Wn=2*截止频率/采样频率。（注意：根据采样定理，采样频率要大于两倍的信号本身最大的频率，才能还原信号。截止频率一定小于信号本身最大的频率，所以Wn一定在0和1之间）。当构造带通滤波器或者带阻滤波器时，Wn为长度为2的列表。

btype : 滤波器类型{‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’},

output : 输出类型{‘ba’, ‘zpk’, ‘sos’},

输出参数：

b，a: IIR滤波器的分子（b）和分母（a）多项式系数向量。output='ba'

z,p,k: IIR滤波器传递函数的零点、极点和系统增益. output= 'zpk'

sos: IIR滤波器的二阶截面表示。output= 'sos'

2.函数的使用

信号滤波中最常用的无非低通滤波、高通滤波和带通滤波。下面简单介绍这三种滤波的使用过程：

(1).高通滤波

#这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除10hz以下频率成分，即截至频率为10hz，则wn=2*10/1000=0.02

from scipy import signal

b, a = signal.butter(8, 0.02, 'highpass')
filtedData = signal.filtfilt(b, a, data)#data为要过滤的信号

(2).低通滤波

#这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除10hz以上频率成分，即截至频率为10hz，则wn=2*10/1000=0.02

from scipy import signal

b, a = signal.butter(8, 0.02, 'lowpass')  
filtedData = signal.filtfilt(b, a, data)       #data为要过滤的信号

 

(3).带通滤波

#这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除10hz以下和400hz以上频率成分，即截至频率为10hz和400hz,则wn1=2*10/1000=0.02,wn2=2*400/1000=0.8。Wn=[0.02,0.8]

from scipy import signal

b, a = signal.butter(8, [0.02,0.8], 'bandpass')
filtedData = signal.filtfilt(b, a, data)   #data为要过滤的信号
--------------------- 
原文：https://blog.csdn.net/weixin_37996604/article/details/82864680 
'''