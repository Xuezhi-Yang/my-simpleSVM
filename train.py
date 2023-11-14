from dataload import dataload
import MySVM as MS
import VaLu

#输入你要训练的数据集(ADNI,PPMI,OCD,FTD,ADNI_fMRI)
X_train,X_test,X_validation,y_train,y_test,y_validation=dataload('PPMI',randomseed=42)



#训练模型，max_iter选择训练代数，kenel选择核函数,可选gamma,c,ceof0，推荐两个ADNI使用sigmoid或rbf，其余使用linear
svm = MS.MySVM(max_iter=1000,kernel='linear')
svm.fit(X_train, y_train)

#显示训练效果
VaLu.VL(svm,X_test,X_validation,y_test,y_validation)

#保存模型
# VaLu.save_model(svm, 'model.pkl')

# # 加载模型并使用
# loaded_svm = VaLu.load_model('model.pkl')
# result = loaded_svm.predict(X_test)