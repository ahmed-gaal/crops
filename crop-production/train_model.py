import pickle

import pandas as pd 
from sklearn.linear_model import LinearRegression

from config import Config

Config.models_path.mkdir(parents=True, exist_ok=True)

x_train = pd.read_csv(str(Config.features_path / 'train_features.csv'))
y_train = pd.read_csv(str(Config.features_path / 'train_target.csv'))

model = LinearRegression()
model = model.fit(x_train,y_train.to_numpy().ravel())

pickle.dump(model, open(str(Config.models_path / 'model.pickle'), 'wb'))