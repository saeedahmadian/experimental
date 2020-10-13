import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import copy

class DataPreparation(object):
    def __init__(self,filename='Smoothed_Data_Cancer2.csv'):
        self.rawDf=pd.read_csv(filename, header='infer')
        self.nonDos_features = ['Age', 'BMI', 'Total_blood_volume_litres_Nadlerformula','PTV']
        self.Dos_features = ['bodyV5_rel', 'bodyV10_rel', 'bodyV15_rel',
                   'bodyV20_rel', 'bodyV25_rel', 'bodyV30_rel', 'bodyV35_rel', 'bodyV40_rel',
                   'bodyV45_rel', 'bodyV50_rel', 'meanbodydose', 'bodyvolume', 'lungV5_rel',
                   'lungV10_rel', 'lungV15_rel', 'lungV20_rel', 'lungV25_rel', 'lungV30_rel',
                   'lungV35_rel', 'lungV40_rel', 'lungV45_rel', 'lungV50_rel', 'meanlungdose',
                   'lungvolume', 'heartV5_rel', 'heartV10_rel', 'heartV15_rel', 'heartV20_rel',
                   'heartV25_rel', 'heartV30_rel', 'heartV35_rel', 'heartV40_rel', 'heartV45_rel',
                   'heartV50_rel', 'meanheartdose', 'heartvolume', 'spleenV5_rel', 'spleenV10_rel',
                   'spleenV15_rel', 'spleenV20_rel', 'spleenV25_rel', 'spleenV30_rel', 'spleenV35_rel',
                   'spleenV40_rel', 'spleenV45_rel', 'spleenV50_rel', 'meanspleendose', 'spleenvolume'
                   ]

        self.nonDos_sparse_features = ['IMRT1Protons0', 'Sex', 'Race', 'Histology',
                           'Location_uppmid_vs_low', 'Location_upp_vs_mid_vs_low', 'Induction_chemo',
                           'CChemotherapy_type']

        self.sequential_features_t0 = ['W0']
        self.sequential_features_t1 = ['W1']
        self.sequential_features_t2 = ['W2']
        self.sequential_features_t3 = ['W3']
        self.sequential_features_t4 = ['W4']
        self.sequential_features_t5 = ['W5']

        self.sequential_features = self.sequential_features_t0+self.sequential_features_t1 +\
                                   self.sequential_features_t2 +self.sequential_features_t3 + \
                                   self.sequential_features_t4 + self.sequential_features_t5
        all_features = self.nonDos_features+ self.Dos_features+self.nonDos_sparse_features+ \
                       self.sequential_features
        self.DF = self.string_float(self.rawDf[all_features])

    def string_float(self,data):
        df = copy.deepcopy(data)
        columns = data.shape[1]
        for col in range(columns):
            tmp = list(map(lambda x: 0 if x == ' ' else float(x), data.iloc[:, col].tolist()))
            median = np.median(tmp)
            df.iloc[:, col] = list(map(lambda x: median if x == 0 else x, tmp))
        return df

    def split_train_test(self,test_size):
        self.nonDos_scaler= MinMaxScaler((0,1)).fit(self.DF[self.nonDos_features].values)
        self.Dos_scaler= MinMaxScaler((0,1)).fit(self.DF[self.Dos_features].values)
        self.nonDos_sparse_scaler= MinMaxScaler((0,1)).fit(self.DF[self.nonDos_sparse_features].values)
        self.seq_scaler= MinMaxScaler((0,1)).fit(self.DF[self.sequential_features].values)
        self.x_nonDos = self.nonDos_scaler.transform(self.DF[self.nonDos_features].values)
        self.x_Dos = self.Dos_scaler.transform(self.DF[self.Dos_features].values)
        self.x_nonDos_sparse= self.DF[self.nonDos_sparse_features].values
        self.x_seq= self.seq_scaler.transform(self.DF[self.sequential_features].values)
        self.x_Dos_train,self.x_Dos_test, self.x_nonDos_train,self.x_nonDos_test,\
        self.x_nonDos_sparse_train,self.x_nonDos_sparse_test,self.x_seq_train,self.x_seq_test=\
        train_test_split(self.x_nonDos,self.x_Dos,self.x_nonDos_sparse,self.x_seq,test_size=test_size,
                         random_state=10)
        return self.x_Dos_train,self.x_Dos_test, self.x_nonDos_train,self.x_nonDos_test,\
        self.x_nonDos_sparse_train,self.x_nonDos_sparse_test,self.x_seq_train,self.x_seq_test





