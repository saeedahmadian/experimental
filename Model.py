import tensorflow as tf
import copy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sn

tf.keras.backend.set_floatx('float64')


class SeqEncoder(tf.keras.Model):
    def __init__(self, enc_layers,enc_units, batch_size):
        super(SeqEncoder, self).__init__()
        self.enc_units = enc_units
        self.enc_layers= enc_layers
        self.batch_size = batch_size
        self.lstm_layers=[]
        if enc_layers < 1:
            raise Exception("Number of layers must be greater than 1")
        else:
            for i in range(enc_layers):
                self.lstm_layers.append(
                    tf.keras.layers.LSTM(self.enc_units,return_sequences=True,
                                         return_state=True,recurrent_initializer='glorot_uniform',
                                         name='encoder_lstm_layer_{}'.format(i)))

    # @tf.function
    def call(self, x, state):
        for layer in self.lstm_layers:
            x, *state = layer(x, initial_state = state)
        return x, state


    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_size, self.enc_units),dtype=tf.float64),
                tf.zeros((self.batch_size, self.enc_units),dtype=tf.float64)]



class Attention(tf.keras.layers.Layer):

    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    # @tf.function
    def call(self, states, onc_out):

        h_state, c_state = states

        h_state_time_axis = tf.expand_dims(h_state, 1)

        score = self.V(tf.nn.tanh(self.W1(h_state_time_axis) + self.W2(onc_out)))

        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * onc_out
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights[:,:,0]


class SeqDecoder(tf.keras.Model):
    def __init__(self, output_size, output_seq_len, dec_units, batch_size):

        super(SeqDecoder, self).__init__()
        self.dec_units = dec_units
        self.batch_size = batch_size
        self.output_seq_len= output_seq_len
        self.inp_layers=[]
        self.lstm_cels=[]
        self.fc=[]
        for i in range(self.output_seq_len):
            self.inp_layers.append(
                tf.keras.layers.Dense(self.dec_units,name='decoder_input_layer_{}'.format(i)))
            self.lstm_cels.append(tf.keras.layers.LSTMCell(self.dec_units,
                                                           recurrent_initializer='glorot_uniform',
                                                           name='decoder_lstm_cel_{}'.format(i)))
            self.fc.append(tf.keras.layers.Dense(output_size,name='decoder_last_layer_{}'.format(i)))


        self.attention = Attention(self.dec_units)

    # @tf.function
    def call(self, states, enc_output):

        self.out_lst = []
        self.att_weights=[]
        for inp,lstm_cel,last_layer in zip(self.inp_layers,self.lstm_cels,self.fc):
            context_vector, attention_weights = self.attention(states, enc_output)
            self.att_weights.append(attention_weights)
            # x= inp(x)
            # x_new = tf.concat([context_vector, x], axis=-1)
            x_new= context_vector
            pred,states= lstm_cel(x_new,states)
            # x = pred
            pred= last_layer(pred)
            self.out_lst.append(pred)

        predictions= tf.stack(self.out_lst)
        probs = tf.stack(self.att_weights)
        predictions= tf.transpose(predictions,[1, 0, 2])
        probs= tf.transpose(probs,[1,0,2])
        return predictions, probs


def string_float(data):
    df = copy.deepcopy(data)
    columns = data.shape[1]
    for col in range(columns):
        tmp = list(map(lambda x: 0 if x == ' ' else float(x), data.iloc[:, col].tolist()))
        median = np.median(tmp)
        df.iloc[:, col] = list(map(lambda x: median if x == 0 else x, tmp))
    return df

def myplot(y_true, y_pred, nrows=5, ncols=5, title='Saeed_CRT1lymphocyte_absolute_count_KuL2'):
    fig, axes = plt.subplots(nrows, ncols, sharex='none', sharey='none', figsize=(20, 20))
    seq_len = y_true.shape[1]
    i = 0
    for row_ax in axes:
        for col_ax in row_ax:
            col_ax.plot(np.arange(seq_len), y_true[i, :], color='darkblue', label='True_value')
            col_ax.plot(np.arange(seq_len), y_pred[i, :], color='darkorange', label='Pred_value')
            col_ax.set_xticks(range(0, 5))
            col_ax.set_xticklabels([str(i) for i in range(1, seq_len + 1)])
            col_ax.set_xlabel('Weeks')
            col_ax.set_ylabel('ALC')
            col_ax.legend()
            i += 1
    fig.savefig('./Figs1/fignew_{}.png'.format(title))


data= pd.read_csv('cancer2.csv',header='infer')
sequential_features_t0 = [
    'CRT0neutrophil_percent', 'CRT0lymphocyte_percent', 'CRT0monocyte_percent', 'CRT0ALC']
sequential_features_t1 = ['CRT1lymphocyte_absolute_count_KuL']
sequential_features_t2 = ['CRT2lymphocyte_absolute_count_KuL']
sequential_features_t3 = ['CRT3lymphocyte_absolute_count_KuL']
sequential_features_t4 = ['CRT4lymphocyte_absolute_count_KuL']
sequential_features_t5 = ['CRT5lymphocyte_absolute_count_KuL']

sequential_features = sequential_features_t1 + sequential_features_t2 + \
                      sequential_features_t3 + sequential_features_t4 + \
                      sequential_features_t5
data_train, data_test = train_test_split(data, test_size=.3, random_state=8)


x_train_init_seq = MinMaxScaler((0, 1)).\
    fit_transform(string_float(data_train[sequential_features_t0]).values)
y_train_sequential = MinMaxScaler((0, 1)).\
    fit_transform(string_float(data_train[sequential_features]).values)

x_test_init_seq = MinMaxScaler((0, 1)).\
    fit_transform(string_float(data_test[sequential_features_t0]).values)
y_test_sequential = MinMaxScaler((0, 1)).\
    fit_transform(string_float(data_test[sequential_features]).values)



units= 8
enc_layers = 1
batch_size= 32
in_len= 1
in_features=x_train_init_seq.shape[1]
out_len = 5
out_size= 1
lr=0.05
Buffer_size= len(x_train_init_seq)
Disp_freq=30

dataset_train = tf.data.Dataset. \
    from_tensor_slices((x_train_init_seq,y_train_sequential)).batch(batch_size,drop_remainder=True)

dataset_test = tf.data.Dataset. \
    from_tensor_slices((x_test_init_seq,y_test_sequential))

# test_enc= SeqEncoder(1,100,2)
# test_dec= SeqDecoder(3,5,100,2)
# enc_out, states= test_enc(tf.random.normal(shape=(2,8,1)),test_enc.initialize_hidden_state())
# predictions, probs=test_dec(enc_out[:, -1, :], states, enc_out)


encoder = SeqEncoder(enc_layers,units,batch_size)
decoder= SeqDecoder(out_size,out_len,units,batch_size)
logdir = 'test'
writer = tf.summary.create_file_writer(logdir)
draw= False
if draw==True:
    with writer.as_default():
        init_state = encoder.initialize_hidden_state()
        x_enc, targ = next(iter(dataset_train))
        x_enc = tf.expand_dims(x_enc, 1)
        targ = tf.expand_dims(targ, 2)
        tf.summary.trace_on(graph=True, profiler=True)
        enc_out, states = encoder(x_enc, init_state)
        tf.summary.trace_export(
            name="Encoder",
            step=0,
            profiler_outdir=logdir)
        tf.summary.trace_on(True, True)
        pred, attention = decoder(enc_out[:, -1, :], states, enc_out)
        tf.summary.trace_export(
            name="Decoder",
            step=0,
            profiler_outdir=logdir)


loss=tf.keras.losses.MeanSquaredError()
optim= tf.keras.optimizers.Adam(learning_rate=lr)

def train_step(x_inp_enc,init_state,y_tar):
    with tf.GradientTape() as g:
        enc_out, states= encoder(x_inp_enc,init_state)
        pred,_ = decoder(x_inp_enc[:,-1,:],states,enc_out)
        L= loss(y_tar,pred)
    vars= encoder.trainable_variables + decoder.trainable_variables
    grads= g.gradient(L,vars)
    optim.apply_gradients(zip(grads,vars))
    return L,vars,grads,pred

def train(Epochs,dataset):
    glob_iter=0
    total_loss=[]
    for epoch in range(Epochs):
        dataset= dataset.shuffle(Buffer_size)
        local_iter=0
        for x_enc,targ in dataset:
            init_state=  encoder.initialize_hidden_state()
            x_enc= tf.expand_dims(x_enc,1)
            targ = tf.expand_dims(targ, 2)
            with writer.as_default():
                batch_loss,vars,grads,preds = train_step(x_enc, init_state, targ)
                tf.summary.scalar('train_loss',batch_loss,glob_iter)
                total_loss.append(batch_loss)
                if glob_iter % Disp_freq == 0:
                    print('Epoch {} , iter {}, loss {}'.format(epoch, local_iter, batch_loss))

            glob_iter+=1
            local_iter+=1
    return total_loss

def evaluate(dataset):
    batch_size=len(dataset)
    dataset= dataset.batch(batch_size)
    for x_enc, targ in dataset:
        init_state = [tf.zeros((batch_size, units),dtype=tf.float64),
                tf.zeros((batch_size, units),dtype=tf.float64)]
        x_enc = tf.expand_dims(x_enc, 1)
        targ= tf.expand_dims(targ,2)
        enc_out, states = encoder(x_enc, init_state)
        pred, probs = decoder(x_enc[:, -1, :], states, enc_out)
        mse_test= loss(targ,pred)
    return pred,targ,probs,mse_test


loss_train= train(20,dataset_train)
preds,targs,probs,mste_test= evaluate(dataset_test)

for i in range(0,len(preds)-25,25):
    myplot(targs[i:i+25,:,0],preds[i:i+25,:,0],title='batch_{}'.format(i//25))



a=1

