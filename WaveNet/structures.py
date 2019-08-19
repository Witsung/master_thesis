from utils import prepare_data, build_wavenet
from pandas import read_csv, DataFrame
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

df = read_csv('data/v_hub_filt.csv')
df = df[['v_hub_filt']]
input_steps = 64
pred_steps = 5
X_train, y_train, X_test, y_test, ss_test = prepare_data(df, input_steps=64, pred_steps=1,
                                                         pred_overlaps=63, train_size=0.8)

dilated_layers = [4, 5, 6]
n_stacks = [1, 2, 3, 4, 5]

param_grid = dict(dilated_layers=dilated_layers, n_stacks=n_stacks)

model = KerasRegressor(build_fn=build_wavenet, epochs=100, verbose=1)

es = EarlyStopping(monitor='val_loss', patience=8, verbose=2)
callbacks = [es]

tscv = TimeSeriesSplit(n_splits=3)
grid = GridSearchCV(estimator=model, cv=tscv, param_grid=param_grid,
                    verbose=2, n_jobs=16,
                    scoring=['neg_mean_absolute_error', 'neg_mean_squared_error'],
                    refit=False, return_train_score=False)

grid_result = grid.fit(X_train, y_train, shuffle=True, callbacks=callbacks,
                       validation_split=0.1)

DataFrame.from_dict(grid_result.cv_results_).to_csv('data/WaveNet_grid_result_3.csv', index=False)
