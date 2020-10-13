from Preprocess import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import copy
from sklearn.manifold import TSNE
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import IterativeImputer

def string_float(data):
    df = copy.deepcopy(data)
    columns = data.shape[1]
    for col in range(columns):
        tmp = list(map(lambda x: 0 if x == ' ' else float(x), data.iloc[:, col].tolist()))
        median = np.median(tmp)
        df.iloc[:, col] = list(map(lambda x: median if x == 0 else x, tmp))
    return df


tf.keras.backend.set_floatx('float64')
#'New_Data_Deleted_Wrong_Trend.csv'
#'Smoothed_Data_Cancer2.csv'
mypipline = Pipeline([('read_data', ReadData(data_dir='./Data', file_name='Smoothed_Data_Cancer2.csv'))
                      #    ,
                      # ('clean_data',CleanData('median')),
                      # ('cat_to_num',CatToNum('ordinal')),
                      # ('Outlier_mitigation',OutlierDetection(threshold=2,name='whisker'))
                      ])

new_data = mypipline.fit_transform(None)

dense_features1 = ['Age', 'BMI', 'Total_blood_volume_litres_Nadlerformula','PTV']
dense_features2 = ['bodyV5_rel', 'bodyV10_rel', 'bodyV15_rel',
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

# new_data_dense_features2=string_float(new_data[dense_features2]).values
# N = 20000
# n_components = 3
# tsne = TSNE(n_components=n_components, perplexity=50, n_iter=N, init='pca',random_state=0,verbose=1, method='exact')
# tsne_results = tsne.fit_transform(new_data_dense_features2)

# new_data=new_data.drop(dense_features2, axis=1)
# new_data["DoseComp1"]=tsne_results[:,0]
# new_data["DoseComp2"]=tsne_results[:,1]
# new_data["DoseComp3"]=tsne_results[:,2]
# dense_features2_new=["DoseComp1", "DoseComp2", "DoseComp3"]

# new_data.to_csv('New_Data_471_tsne_20K_3c.csv', index=False)

data_train, data_test = train_test_split(new_data, test_size=.2, random_state=8)

data_test.to_csv('Smoothed_ntr_Real_Testset_2.csv', index=False)

y_train_class = data_train.pop('G4RIL').values
y_train_reg = data_train.pop('CRT_ALCnadir')

y_test_class = data_test.pop('G4RIL').values
y_test_reg = data_test.pop('CRT_ALCnadir')

# dense_features = dense_features1 + dense_features2_new
dense_features = dense_features1 + dense_features2

sparse_features = ['IMRT1Protons0', 'Sex', 'Race', 'Histology',
                   'Location_uppmid_vs_low', 'Location_upp_vs_mid_vs_low', 'Induction_chemo',
                   'CChemotherapy_type']
sequential_features_t0 = ['W0']
# sequential_features_t0 = ['CRT0ALC']

# sequential_features_t1 = ['CRT1lymphocyte_absolute_count_KuL']
# sequential_features_t2 = ['CRT2lymphocyte_absolute_count_KuL']
# sequential_features_t3 = ['CRT3lymphocyte_absolute_count_KuL']
# sequential_features_t4 = ['CRT4lymphocyte_absolute_count_KuL']
# sequential_features_t5 = ['CRT5lymphocyte_absolute_count_KuL']

sequential_features_t1 = ['W1']
sequential_features_t2 = ['W2']
sequential_features_t3 = ['W3']
sequential_features_t4 = ['W4']
sequential_features_t5 = ['W5']

sequential_features = sequential_features_t1 + sequential_features_t2 + \
                      sequential_features_t3 + sequential_features_t4 + \
                      sequential_features_t5

seq_features=sequential_features_t0+sequential_features_t1 + sequential_features_t2 + \
                      sequential_features_t3 + sequential_features_t4 + \
                      sequential_features_t5

class saba_class(tf.keras.models.Model):
    def __init__(self, hidd_size=20, lstm_size=20, max_seq=5, **kwargs):
        super(saba_class, self).__init__(**kwargs)
        self.max_seq = max_seq
        self.hidd_size = hidd_size
        self.lstm_size = lstm_size
        self.dense_layer = tf.keras.layers.Dense(units=20, activation='relu', name='Dense_input', dtype=tf.float64)
        self.sparse_layer = tf.keras.layers.Dense(units=20,
                                                  kernel_regularizer=tf.keras.regularizers.l1(.2),
                                                  name='sparse_input', dtype=tf.float64)
        self.init_seq = tf.keras.layers.Dense(units=10, activation='relu', name='initial_sequence', dtype=tf.float64)
        self.combine_layer = tf.keras.layers.Dense(units=self.hidd_size, activation='relu',
                                                   name='combined_layer', dtype=tf.float64)
        self.lstm_layer_list = [tf.keras.layers.LSTM(units=self.lstm_size, return_sequences=True,
                                                     return_state=True, name='lstm_cell_{}'.format(i), dtype=tf.float64)
                                for i in range(self.max_seq)]

        # self.final_lstm= tf.keras.layers.LSTM(1)

    # @tf.function(input_signature=[tf.TensorSpec(shape=[None,53],dtype=tf.float64,name='input_dense'),
    #                               tf.TensorSpec(shape=[None,8],dtype=tf.float64,name='input_sparse'),
    #                               tf.TensorSpec(shape=[None,4],dtype=tf.float64,name='input_init_seq'),
    #                               tf.TensorSpec(shape=[None,1],dtype=tf.float64,name='initial_state_1'),
    #                               tf.TensorSpec(shape=[None, 1], dtype=tf.float64, name='initial_state_2')
    #                               ])
    def call(self, x_dense, x_sparse, x_init_seq, initial_state1, initial_state2):
        x_dense = self.dense_layer(x_dense)
        x_sparse = self.sparse_layer(x_sparse)
        x_init_seq = self.init_seq(x_init_seq)
        x_proc = self.combine_layer(tf.concat([x_dense, x_sparse, x_init_seq], axis=-1))
        out = tf.expand_dims(x_proc, 1)
        seq_output = []
        i = 0
        init_states = [initial_state1, initial_state2]
        for lstm in self.lstm_layer_list:
            out, hiddent_state, cell_state = lstm(out, initial_state=init_states)
            seq_output.append(out)
            init_states = [hiddent_state, cell_state]
            i += 1
        # out= self.last_layer(tf.reshape(out,shape=[-1,self.lstm_size]))
        return out, seq_output

    def get_config(self):
        config = super(saba_class, self).get_config().copy()
        config.update({
            'maximum_length_seq': self.max_seq,
            'hidden_size': self.hidd_size,
            'LSTM_size': self.lstm_size
        })
        return config


class SaeedClass(tf.keras.models.Model):
    def __init__(self, hidd_size=20, latent_size=20, lstm_size=20, max_seq=5, **kwargs):
        super(SaeedClass, self).__init__(**kwargs)
        self.max_seq = max_seq
        self.hidd_size = hidd_size
        self.lstm_size = lstm_size
        self.latent_size = latent_size

        self.NDF_layer = tf.keras.layers.Dense(units=50, activation='relu', name='Dense_input', dtype=tf.float64)

        self.NDF_layer_1 = tf.keras.layers.Dense(units=20, activation='relu', name='Dense_input_1', dtype=tf.float64)

        self.NDSF_layer = tf.keras.layers.Dense(units=20, kernel_regularizer=tf.keras.regularizers.l1(.2),
                                                name='sparse_input', dtype=tf.float64)
        self.drop_out = tf.keras.layers.Dropout(rate=.2)
        self.NDSF_layer_1 = tf.keras.layers.Dense(units=20, kernel_regularizer=tf.keras.regularizers.l1(.2),
                                                  name='sparse_input_1', dtype=tf.float64)

        self.combine_layer = tf.keras.layers.Dense(units=self.hidd_size, activation='relu',
                                                   name='combined_layer', dtype=tf.float64)

        self.init_seq = tf.keras.layers.Dense(units=10, activation='relu', name='initial_sequence', dtype=tf.float64)
        self.latent = tf.keras.layers.Dense(units=self.latent_size, activation='relu', name='latent_layer')
        self.lstm_layer_list = [tf.keras.layers.LSTM(units=self.lstm_size, return_sequences=True,
                                                     return_state=True, name='lstm_cell_{}'.format(i), dtype=tf.float64)
                                for i in range(self.max_seq)]

        # self.final_lstm= tf.keras.layers.LSTM(1)

    # @tf.function(input_signature=[tf.TensorSpec(shape=[None,53],dtype=tf.float64,name='input_dense'),
    #                               tf.TensorSpec(shape=[None,8],dtype=tf.float64,name='input_sparse'),
    #                               tf.TensorSpec(shape=[None,4],dtype=tf.float64,name='input_init_seq'),
    #                               tf.TensorSpec(shape=[None,1],dtype=tf.float64,name='initial_state_1'),
    #                               tf.TensorSpec(shape=[None, 1], dtype=tf.float64, name='initial_state_2')
    #                               ])
    def call(self, x_dense, x_sparse, x_init_seq, initial_state1, initial_state2):
        x_NDF = self.NDF_layer(x_dense)
        x_NDF_1 = self.NDF_layer_1(x_NDF)

        x_NDSF = self.NDSF_layer(x_sparse)
        x_NDSF_drop_out = self.drop_out(x_NDSF)
        x_NDSF_1 = self.NDSF_layer_1(x_NDSF_drop_out)

        ### Flatten
        x_combine = self.combine_layer(tf.concat([x_NDF_1, x_NDSF_1], axis=-1))

        ##########
        x_init_seq = self.init_seq(x_init_seq)
        x_latent = self.latent(tf.concat([x_init_seq, x_combine], axis=-1))
        ###########################
        out = tf.expand_dims(x_latent, 1)
        seq_output = []
        i = 0
        init_states = [initial_state1, initial_state2]
        for lstm in self.lstm_layer_list:
            out, hiddent_state, cell_state = lstm(out, initial_state=init_states)
            seq_output.append(out)
            init_states = [hiddent_state, cell_state]
            i += 1
        # out= self.last_layer(tf.reshape(out,shape=[-1,self.lstm_size]))
        return out, seq_output

    def get_config(self):
        config = super(SaeedClass, self).get_config().copy()
        config.update({
            'maximum_length_seq': self.max_seq,
            'hidden_size': self.hidd_size,
            'LSTM_size': self.lstm_size
        })
        return config





MAX_SEQ = 5
epochs = 945 #750 #500
batch_size = 32
NUM_SEQ_Features = 1
optim = tf.keras.optimizers.Adam(learning_rate=1e-3, name='adam_optim')
model = SaeedClass(lstm_size=NUM_SEQ_Features)

scaler = MinMaxScaler((0, 1)).fit(string_float(data_train[seq_features]).values)

x_train_dense = MinMaxScaler((0,1)).fit_transform(string_float(data_train[dense_features]).values)
# x_train_dense = MinMaxScaler((0, 1)).fit_transform(data_dense_train)
x_train_sparse = MinMaxScaler((0, 1)).fit_transform(string_float(data_train[sparse_features]).values)

seq_train=scaler.transform(string_float(data_train[seq_features]).values)
x_train_init_seq = seq_train[:,0].reshape((-1,1))
y_train_sequential = seq_train[:,1:]

y_train_class

dataset = tf.data.Dataset. \
    from_tensor_slices((x_train_dense, x_train_sparse, x_train_init_seq,
                        y_train_sequential, y_train_class)).batch(batch_size)

checkpoint_dir = './check_points'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optim,
                                 model=model)

loss_pred = tf.keras.losses.MeanSquaredError(name='mse')
loss_class = tf.keras.losses.BinaryCrossentropy(name='cross_entropy')


def train_step(x_dense, x_sparse, x_init_seq, init_state1, init_state2, seq_target,
               class_target):
    loss_total = 0
    with tf.GradientTape() as tape:
        y_label, output_sequence = model(x_dense, x_sparse, x_init_seq, init_state1, init_state2)
        for i, out in enumerate(output_sequence):
            y_targ = seq_target[:, i]
            loss_total += loss_pred(y_targ, tf.reshape(out, shape=[-1, NUM_SEQ_Features]))
        # loss_total+=loss_class(class_target,y_label)

    gradients = tape.gradient(loss_total, model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))

    return loss_total


def myplot(y_true, y_pred, nrows=5, ncols=5, title='Saeed_CRT1lymphocyte_absolute_count_KuL2'):
    fig, axes = plt.subplots(nrows, ncols, sharex='none', sharey='none', figsize=(20, 20))
    seq_len = y_true.shape[1]
    i = 0
    for row_ax in axes:
        for col_ax in row_ax:
            col_ax.plot(np.arange(seq_len), y_true[i, :], color='darkblue', label='True_value')
            col_ax.plot(np.arange(seq_len), y_pred[i, :], color='darkorange', label='Pred_value')
            col_ax.set_xticks(range(0, 6))
            col_ax.set_xticklabels([str(i) for i in range(0, seq_len)])
            col_ax.set_xlabel('Weeks')
            col_ax.set_ylabel('ALC')
            col_ax.legend()
            i += 1
    fig.savefig('./Figs1/fignew_{}.png'.format(title))


def myplot2(y_true, nrows=5, ncols=5, title='Saeed_CRT1lymphocyte_absolute_count_KuL'):
    fig, axes = plt.subplots(nrows, ncols, sharex='none', sharey='none', figsize=(20, 20))
    seq_len = y_true.shape[1]
    i = 0
    for row_ax in axes:
        for col_ax in row_ax:
            col_ax.plot(np.arange(seq_len), y_true[i, :], color='darkblue', label='True_value')
            # col_ax.plot(np.arange(seq_len),y_pred[i,:],color='darkorange',label='Pred_value')
            col_ax.legend()
            i += 1
    fig.savefig('./Figs1/fig_truenew_{}.png'.format(title))


df2 = data_test[['CRT1lymphocyte_absolute_count_KuL', 'CRT2lymphocyte_absolute_count_KuL',
                 'CRT3lymphocyte_absolute_count_KuL', 'CRT4lymphocyte_absolute_count_KuL',
                 'CRT5lymphocyte_absolute_count_KuL']]
df2.columns = ['W1', 'W2', 'W3', 'W4', 'W5']
# myplot2( df2.values)

train_loss = []
for epoch in range(epochs):
    print('epoch {}/{} starts...'.format(epoch, epochs))
    dataset = dataset.shuffle(600)
    i = 0
    for x_batch_dense, x_batch_sparse, x_batch_init_seq, y_batch_sequential, y_batch_class in dataset:
        current_batch = x_batch_dense.shape[0]
        initial_state1 = tf.zeros((current_batch, NUM_SEQ_Features), dtype=tf.float64)
        initial_state2 = tf.zeros((current_batch, NUM_SEQ_Features), dtype=tf.float64)
        # initial_state= [tf.zeros((current_batch,NUM_SEQ_Features),dtype=tf.float64),tf.zeros((current_batch,NUM_SEQ_Features),dtype=tf.float64)]
        batch_loss = train_step(x_batch_dense, x_batch_sparse,
                                x_batch_init_seq, initial_state1, initial_state2,
                                y_batch_sequential, y_batch_class)
        train_loss.append(batch_loss)
        #
        # if i % 10 ==0:
        #     checkpoint.save(file_prefix=checkpoint_prefix)
        #     model.save_weights('manual_checkpoint/mymodel-{}-ckpt'.format(i))
        #     print('epoch {}/{} iter {} loss is {}'.format(epoch,epochs,i,batch_loss))
        # loss.append(batch_loss)
        i += 1

    # print('Loss value for epoch {} is {}'.format(epoch,batch_loss))
test_mode = True
if test_mode == True:

    x_test_dense = MinMaxScaler((0, 1)).fit_transform(string_float(data_test[dense_features]).values)
    # x_test_dense = MinMaxScaler((0, 1)).fit_transform(data_dense_test)
    x_test_sparse = MinMaxScaler((0, 1)).fit_transform(string_float(data_test[sparse_features]).values)
    # x_test_init_seq = scaler.transform(string_float(data_test[sequential_features_t0]).values)
    # y_test_sequential = MinMaxScaler((0, 1)).fit_transform(string_float(data_test[sequential_features]).values)
    # y_test_sequential = scaler.transform(string_float(data_test[sequential_features]).values)
    seq_test = scaler.transform(string_float(data_test[seq_features]).values)
    x_test_init_seq = seq_test[:, 0].reshape((-1, 1))
    y_test_sequential = seq_test[:, 1:]

    current_batch = x_test_dense.shape[0]
    initial_state1 = tf.zeros((current_batch, NUM_SEQ_Features), dtype=tf.float64)
    initial_state2 = tf.zeros((current_batch, NUM_SEQ_Features), dtype=tf.float64)
    # init_state= [tf.zeros((current_batch,NUM_SEQ_Features),dtype=tf.float64),tf.zeros((current_batch,NUM_SEQ_Features),dtype=tf.float64)]
    y_label, output_sequence = model(x_test_dense, x_test_sparse, x_test_init_seq, initial_state1, initial_state2)
    RMSE = []
    for i, out in enumerate(output_sequence):
        y_targ = y_test_sequential[:, i]
        out_ = tf.reshape(out, [-1, NUM_SEQ_Features])
        RMSE.append(tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_targ, out_)))
    output_sequence_ = [tf.reshape(out, [-1, NUM_SEQ_Features]) for out in output_sequence]
    saba= tf.stack(output_sequence_)
    saba= tf.transpose(saba,(1,0,2))
    dimension= saba.shape
    saba= tf.reshape(saba,(dimension[0],dimension[1]))
    # output_sequence_1= tf.concat([tf.expand_dims(out[:,0],-1) for out in output_sequence_],axis=-1)
    output_sequence_all=tf.concat([seq_test[:, 0].reshape((-1,1)), saba], axis=1)
    yy = tf.concat(output_sequence_all, 1)
    # output_sequence_2 = tf.concat([tf.expand_dims(out[:,1],-1) for out in output_sequence_],axis=-1)
    # y_test_sequential_1= y_test_sequential[:,[i for i in range(0,5,1) ]]
    # y_test_sequential_2 = y_test_sequential[:, [i for i in range(1, 5, 1)]]
    print('Start to plot figures')
    myplot(scaler.inverse_transform(seq_test), scaler.inverse_transform(yy), 5, 5,
           '_Smoothed_ntr_1000_2_CRT1lymphocyte_absolute_count_KuL')
    # myplot(y_test_sequential_2,output_sequence_2,5,5,'CRT1lymphocyte_percent')

    # fig2, axes2 = plt.subplots(3, 3, 'none', 'none', figsize=(20, 15))
    # for i in range(3):
    #     for j in range(3):
    #         axes2[i][j].plot([int(i) for i in range(1, 6)], df2.values[i + 3 * j, :], '--r',
    #                         label='true data')
    #
    #         axes2[i][j].legend()
    #         axes2[i][j].set_xlabel('week')
    #         axes2[i][j].set_ylabel('ALC')

    a = 1
# data_test.to_csv('Real_test_471_3.csv', index=False)

predicted=pd.DataFrame(scaler.inverse_transform(yy))
predicted.to_csv('Smoothed_ntr_Predicted_test_2.csv', index=False)
real=pd.DataFrame(scaler.inverse_transform(seq_test))
real.to_csv('Smoothed_ntr_real_test_2.csv', index=False)
# myplot(scaler.inverse_transform(y_test_sequential)[125:150,:],scaler.inverse_transform(yy)[125:150,:],5,5,'125_150_min_max')

plt.plot(train_loss)

loss_pred(seq_test,yy)
loss_pred(scaler.inverse_transform(seq_test),scaler.inverse_transform(yy))


#####################
loss = tf.keras.losses.mean_absolute_error(y_true, y_pred)
loss = tf.keras.losses.MAE(y_true, y_pred)

mape = tf.keras.losses.MeanAbsolutePercentageError()
mape(y_true, y_pred).numpy()









