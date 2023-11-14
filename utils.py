import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(0)

data = scipy.io.loadmat('./dataset/PPMI.mat')
PD = data['PD']
NC = data['NC']
X = np.concatenate((PD, NC), axis=0)
y = np.concatenate((np.zeros(PD.shape[0]), np.ones(NC.shape[0]) * 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)  # 使用训练集拟合标准化器
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 标准化数据集
print(X_test)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)

# 进行预测
y_pred = svm_model.predict(X_test_scaled)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率：{:.2%}".format(accuracy))



