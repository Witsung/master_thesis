from pandas import read_csv
from utils import prepare_data, build_lstm_h_v
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

df = read_csv('data/v_hub_filt.csv')
input_steps = 32
pred_steps = 5
X_train, y_train, X_test, y_test, ss_test = prepare_data(df, input_steps=input_steps,
                                                         pred_steps=pred_steps,
                                                         train_size=0.8)

for i in range(16):

    model = build_lstm_h_v(input_steps=32, n_features=2, output_steps=5, n_neurons=8,
                           recurrent_dropout=0.0, lr_scale=2.934966883)

    mc = ModelCheckpoint('LSTM/LSTM_final_{}.h5'.format(i), monitor='val_loss', verbose=1, save_best_only=True)
    es = EarlyStopping(monitor='val_loss', patience=8, verbose=2)
    tb = TensorBoard(log_dir="LSTM/logs/LSTM_final_{}".format(i))
    callbacks = [mc, es, tb]

    model.fit(X_train, y_train, shuffle=True, verbose=1, callbacks=callbacks, batch_size=32,
              epochs=100, validation_split=0.1)