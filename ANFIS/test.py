from ANFIS import ANFIS
from utils import prepare_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

df = pd.read_csv('data/v_hub_filt.csv')

df = df[['v_hub_filt']].diff().fillna(method='backfill')
input_steps = 8
pred_steps = 1
X_train, y_train, X_test, y_test, ss_test = prepare_data(df, input_steps=input_steps,
                                                         pred_steps=pred_steps,
                                                         train_size=0.8)
# reshape X_train to keep the order of both features.
# X_train = np.array([np.concatenate([X_train[i,:,0], X_train[i,:,1]]) for i in range(X_train.shape[0])])

model = ANFIS(n_inputs=8, n_rules=2, batch_size=64, learning_rate=0.009, mf='gaussian')
history = model.fit(X=X_train, y=y_train, epochs=300, validation_split=0.1, patience=10)

fig = figure(num=None, figsize=(20, 6))
plt.plot(history[:, 0], linewidth=2)
plt.plot(history[:, 1], linewidth=2)
plt.legend(['train_loss', 'val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')

preds = model.predict(X=X_test, y=y_test, pred_steps=5)
np.save('ANFIS/ANFIS_preds_2.npy', preds)
plt.show()