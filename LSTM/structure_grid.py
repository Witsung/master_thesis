from pandas import read_csv, DataFrame
from utils import prepare_data, build_lstm_s
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, CSVLogger
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

df = read_csv('data/v_hub_filt.csv')
input_steps = 32
pred_steps = 5
X_train, y_train, X_test, y_test, ss_test = prepare_data(df, input_steps=input_steps,
                                                         pred_steps=pred_steps, train_size=0.8)

n_neurons = [0, 24, 32, 40]
neurons = [[h1, h2, h3] for h1 in n_neurons for h2 in n_neurons for h3 in n_neurons
           if (h1 != 0) and ((h2 != 0) or (h2 == 0 and h3 == 0))]
param_grid = dict(neurons=neurons)

model = KerasRegressor(build_fn=build_lstm_s, epochs=100, verbose=1)

es = EarlyStopping(monitor='val_loss', patience=8, verbose=2)
csv_log = CSVLogger('data/grid_logs.csv', separator=',', append=True)
callbacks = [es, csv_log]

tscv = TimeSeriesSplit(n_splits=3)
grid = GridSearchCV(estimator=model, cv=tscv, param_grid=param_grid,
                    verbose=2, n_jobs=16,
                    scoring=['neg_mean_absolute_error', 'neg_mean_squared_error'],
                    refit='neg_mean_squared_error')

grid_result = grid.fit(X_train, y_train, shuffle=True, callbacks=callbacks,
                       validation_split=0.1)

DataFrame.from_dict(grid_result.cv_results_).to_csv('data/LSTM_grid_result_final.csv', index=False)
