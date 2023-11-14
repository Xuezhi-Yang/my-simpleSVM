import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def dataload(dataname,randomseed=42):
    k=0
    if dataname=='ADNI':
        data = scipy.io.loadmat('./dataset/ADNI.mat')
        AD = data['AD']
        MCI = data['MCI']
        MCIp = data['MCIp']
        MCIn = data['MCIn']
        NC = data['NC']
        X = np.concatenate((AD, MCI, MCIp, MCIn, NC), axis=0)
        y = np.concatenate(
            (np.zeros(AD.shape[0]), np.ones(MCI.shape[0]), np.ones(MCIp.shape[0]) * 2, np.ones(MCIn.shape[0]) * 3,
             np.ones(NC.shape[0]) * 4))
        k=1
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=randomseed)
        X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=randomseed)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif dataname=='OCD':
        data = scipy.io.loadmat('./dataset/OCD_90_200_fMRI.mat')
        OCD = data['OCD']
        NC = data['NC']
        # Reshape data to 2D
        X = np.concatenate((OCD, NC), axis=0)
        # X = X.reshape(X.shape[0], -1)  # Reshape to (samples, features)
        X = X.reshape((len(X), np.prod(X.shape[1:])))
        # Update labels
        y = np.concatenate((np.ones(OCD.shape[0]) * -1, np.ones(NC.shape[0]) * 1))
        # Split data into train, validation, and test sets
        k = 1
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=randomseed)
        X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=randomseed)
        # Scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_validation = scaler.transform(X_validation)
        X_test = scaler.transform(X_test)
    elif dataname=='PPMI':
        data = scipy.io.loadmat('./dataset/PPMI.mat')
        PD = data['PD']
        NC = data['NC']
        X = np.concatenate((PD, NC), axis=0)
        y = np.concatenate((np.ones(PD.shape[0]) * -1, np.ones(NC.shape[0]) * 1))
        k = 1
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=randomseed)
        X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=randomseed)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif dataname=='ADNI_fMRI':
        data = scipy.io.loadmat('./dataset/ADNI_90_120_fMRI.mat')
        AD = data['AD']
        EMCI = data['EMCI']
        LMCI = data['LMCI']
        NC = data['NC']
        X = np.concatenate((AD, EMCI, LMCI, NC), axis=0)
        X = X.reshape((len(X), np.prod(X.shape[1:])))
        y = np.concatenate(
            (np.zeros(AD.shape[0]), np.ones(EMCI.shape[0]), np.ones(LMCI.shape[0]) * 2,np.ones(NC.shape[0]) * 3))
        k = 1
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=randomseed)
        X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=randomseed)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif dataname=='FTD':
        data = scipy.io.loadmat('./dataset/FTD_90_200_fMRI.mat')
        FTD = data['FTD']
        NC = data['NC']
        # Reshape data to 2D
        X = np.concatenate((FTD, NC), axis=0)
        # X = X.reshape(X.shape[0], -1)  # Reshape to (samples, features)
        X = X.reshape((len(X), np.prod(X.shape[1:])))
        # Update labels
        y = np.concatenate((np.ones(FTD.shape[0]) * -1, np.ones(NC.shape[0]) * 1))
        # Split data into train, validation, and test sets
        k = 1


    if k==1:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=randomseed)
        X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=randomseed)
        # Scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_validation = scaler.transform(X_validation)
        X_test = scaler.transform(X_test)
        return X_train, X_test, X_validation, y_train, y_test, y_validation
    else:
        print('wrong dataname,try again')