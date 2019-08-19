from sklearn.preprocessing import StandardScaler
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

plt.style.use('ggplot')

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, Activation, Dropout, Lambda, Multiply, Add


def prepare_data(df, input_steps=10, pred_steps=5, pred_overlaps=27, train_size=0.8):
    '''The target should be the first column'''
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
    X_test = np.array([scaled_test[i:i + input_steps] for i in range(test_sample_size)])
    y_test = np.array([scaled_test[i + input_steps:i + input_steps + pred_steps][:, 0]
                       for i in range(test_sample_size)])

    y_train = y_train.reshape(-1, pred_steps, 1)
    y_train = np.concatenate([X_train[:, - pred_overlaps:, 0].reshape(-1, pred_overlaps, 1),
                              y_train], axis=1)
    #     X_train = np.concatenate([X_train, y_train[:,-pred_steps:-1,:]], axis=1)

    return X_train, y_train, X_test, y_test, ss_test


def make_pred_plot(df, target_col_name='real', pred_start=0, pred_end=50, pred_interval=2,
                   xlim_low=0, xlim_up=200, get_errors=True, scale=True):
    '''
    get a df has ['real', 's1', 's2', ...] in columns to make plot
    '''

    pred_steps = len(df.drop(columns=target_col_name).columns)
    print("pred_steps: {}".format(pred_steps))

    def evaluate_pred(df, target_col_name):

        real = np.array([df.real[i:pred_steps + i].values for i in range(len(df) - pred_steps + 1)])

        preds = df.drop(columns=target_col_name).values[:len(df) - pred_steps + 1, :]
        errors = real - preds

        rmse = (np.mean(errors ** 2) * pred_steps) ** (1 / 2)
        mae = np.mean(abs(errors)) * pred_steps
        print('RMSE: {}'.format(rmse))
        print('MAE: {}'.format(mae))
        return errors

    errors = evaluate_pred(df, target_col_name=target_col_name)

    if scale:
        ss = StandardScaler()
        df = pd.DataFrame(ss.fit_transform(df), columns=df.columns)

    fig = figure(num=None, figsize=(20, 6))
    ax = fig.add_subplot(111)
    ax.plot(df[target_col_name], 'g')

    for pred_count in range(pred_start, pred_end, pred_interval):
        ax.plot([None for i in range(pred_count)] + list(df.drop(columns=target_col_name).iloc[pred_count, :].values),
                'r--.', linewidth=0.5)

    ax.legend(['real', 'predicted'], prop={'size': 15})
    ax.set_xlim(xlim_low, xlim_up)
    ax.set_xlabel('Step')
    if scale:
        ax.set_ylabel('Scaled_wind_speed')
    else:
        ax.set_ylabel('Wind_speed')
    plt.show()
    if get_errors:
        return fig, ax, errors
    else:
        return fig, ax


def build_wavenet(n_filters=32, filter_width=2, dilated_layers=5, n_stacks=1, n_features=1, output_step=64):
    # Complex WaveNet
    # convolutional operation parameters
    tf.reset_default_graph()
    dilation_rates = [2 ** i for i in range(dilated_layers)] * n_stacks

    # define an input history series and pass it through a stack of dilated causal convolution blocks.
    history_seq = Input(shape=(None, n_features))
    x = history_seq

    skips = []
    for dilation_rate in dilation_rates:
        # preprocessing - equivalent to time-distributed dense
        x = Conv1D(16, 1, padding='same', activation='relu')(x)

        # filter convolution
        x_f = Conv1D(filters=n_filters,
                     kernel_size=filter_width,
                     padding='causal',
                     dilation_rate=dilation_rate)(x)

        # gating convolution
        x_g = Conv1D(filters=n_filters,
                     kernel_size=filter_width,
                     padding='causal',
                     dilation_rate=dilation_rate)(x)

        # multiply filter and gating branches
        z = Multiply()([Activation('tanh')(x_f),
                        Activation('sigmoid')(x_g)])

        # postprocessing - equivalent to time-distributed dense
        z = Conv1D(16, 1, padding='same', activation='relu')(z)

        # residual connection
        x = Add()([x, z])

        # collect skip connections
        skips.append(z)

    # add all skip connection outputs
    out = Activation('relu')(Add()(skips))

    # final time-distributed dense layers
    out = Conv1D(128, 1, padding='same')(out)
    out = Activation('relu')(out)
    out = Dropout(.2)(out)
    out = Conv1D(1, 1, padding='same')(out)

    # extract the last 5 time steps as the training target
    def slice(x, seq_length):
        return x[:, -seq_length:, :]

    pred_seq_train = Lambda(slice, arguments={'seq_length': output_step})(out)
    model = Model(history_seq, pred_seq_train)
    model.compile(optimizer='adam', loss='mse')
    # model.summary()

    return model
