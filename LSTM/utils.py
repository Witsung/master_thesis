from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras import backend
from keras.regularizers import l1_l2
from keras.optimizers import Adam


def prepare_data(df, input_steps=32, pred_steps=5, train_size=0.8):
    split_point = int(len(df) * train_size)
    test_start_index = split_point - input_steps
    ss_train = StandardScaler()
    ss_test = StandardScaler()
    scaled_train = ss_train.fit_transform(df.iloc[:split_point])
    scaled_test = ss_test.fit_transform(df.iloc[test_start_index:])

    train_sample_size = len(scaled_train) - input_steps - pred_steps + 1
    test_sample_size = len(scaled_test) - input_steps - pred_steps + 1
    X_train = np.array([scaled_train[i:i + input_steps] for i in range(train_sample_size)])
    y_train = np.array([scaled_train[i + input_steps:i + input_steps + pred_steps][:, 0]
                        for i in range(train_sample_size)])
    X_test = np.array([scaled_test[i:i + input_steps] for i in range(len(scaled_test) - input_steps)])
    y_test = np.array([scaled_test[i + input_steps:i + input_steps + pred_steps][:, 0]
                       for i in range(test_sample_size)])

    return X_train, y_train, X_test, y_test, ss_test


def build_lstm_s(neurons=[32, 0, 0], input_steps=32, n_features=2, output_steps=5):
    '''build lstm structures for GridsearchCV'''
    backend.clear_session()
    n_layers = sum([n > 0 for n in neurons])
    model = Sequential()
    if n_layers == 1:
        model.add(LSTM(neurons[0], input_shape=(input_steps, n_features)))
    else:
        for l in range(n_layers):
            if l == 0:
                model.add(LSTM(neurons[l],
                               input_shape=(input_steps, n_features),
                               return_sequences=True))
            elif l + 1 == n_layers:
                model.add(LSTM(neurons[l], return_sequences=False))
            else:
                model.add(LSTM(neurons[l], return_sequences=True))
    model.add(Dense(output_steps))
    model.compile(optimizer='adam', loss=['mse'])

    return model


def build_lstm_h(input_steps=32, n_features=2, output_steps=5, dropout=0.0,
                 recurrent_dropout=0.0, lr_scale=3.0,
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'):
    backend.clear_session()
    model = Sequential()
    model.add(LSTM(24, input_shape=(input_steps, n_features), return_sequences=True,
                   # dropout=dropout, recurrent_dropout=recurrent_dropout,
                   # kernel_regularizer=l1_l2(l1=l1, l2=l2), recurrent_regularizer=l1_l2(l1=l1, l2=l2),
                   kernel_initializer=kernel_initializer,
                   recurrent_initializer=recurrent_initializer))
    model.add(LSTM(32, return_sequences=True,
                   dropout=dropout, recurrent_dropout=recurrent_dropout,
                   # kernel_regularizer=l1_l2(l1=l1, l2=l2), recurrent_regularizer=l1_l2(l1=l1, l2=l2),
                   kernel_initializer=kernel_initializer,
                   recurrent_initializer=recurrent_initializer))
    model.add(LSTM(24, return_sequences=False,
                   dropout=dropout, recurrent_dropout=recurrent_dropout,
                   # kernel_regularizer=l1_l2(l1=l1, l2=l2), recurrent_regularizer=l1_l2(l1=l1, l2=l2),
                   kernel_initializer=kernel_initializer,
                   recurrent_initializer=recurrent_initializer))
    model.add(Dense(output_steps))
    lr = 10 ** (- lr_scale)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='mse')
    return model


def build_lstm_h_v(input_steps=32, n_features=2, output_steps=5, n_neurons=8,
                   recurrent_dropout=0.0, lr_scale=3.0,
                   kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'):
    backend.clear_session()
    model = Sequential()
    model.add(LSTM(n_neurons, input_shape=(input_steps, n_features), return_sequences=False,
                   recurrent_dropout=recurrent_dropout,
                   kernel_initializer=kernel_initializer,
                   recurrent_initializer=recurrent_initializer))
    model.add(Dense(output_steps))
    lr = 10 ** (- lr_scale)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='mse')
    return model
