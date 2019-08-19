from pandas import read_csv, DataFrame
from utils import prepare_data, build_lstm_h
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy.stats import uniform

df = read_csv('data/v_hub_filt.csv')
input_steps = 32
pred_steps = 5
X_train, y_train, X_test, y_test, ss_test = prepare_data(df, input_steps=input_steps,
                                                         pred_steps=pred_steps, train_size=0.8)

param_grid = dict(
    dropout=uniform(0.0, 0.6),
    recurrent_dropout=uniform(0.0, 0.6),
    batch_size=[2 ** i for i in range(5, 10)],
    lr_scale=uniform(2.5, 1.5)
)

model = KerasRegressor(build_fn=build_lstm_h, epochs=100, verbose=1)

es = EarlyStopping(monitor='val_loss', patience=8, verbose=2)
callbacks = [es]

tscv = TimeSeriesSplit(n_splits=3)
random = RandomizedSearchCV(estimator=model, cv=tscv, param_distributions=param_grid,
                            verbose=3, n_iter=64, n_jobs=16,
                            scoring=['neg_mean_absolute_error', 'neg_mean_squared_error'],
                            refit='neg_mean_squared_error')

random_result = random.fit(X_train, y_train, shuffle=True, callbacks=callbacks,
                           validation_split=0.1)

DataFrame.from_dict(random_result.cv_results_).to_csv('data/random_result_final_nwi.csv', index=False)
