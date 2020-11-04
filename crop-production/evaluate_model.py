import json
import pickle
import math
import pandas as pd 
from sklearn.metrics import mean_squared_error, r2_score

from config import Config

x_test = pd.read_csv(str(Config.features_path / 'test_features.csv'))
y_test = pd.read_csv(str(Config.features_path / 'test_target.csv'))

model = pickle.load(open(str(Config.models_path / 'model.pickle'), 'rb'))

r_squared = model.score(x_test, y_test)

y_pred = model.predict(x_test)

rmse = math.sqrt(mean_squared_error(y_test, y_pred))

with open(str(Config.metrics_file_path), 'w') as outfile:
    json.dump(dict(r_squared = r_squared, rmse = rmse), outfile)