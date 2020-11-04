import pandas as pd 
from config import Config

Config.features_path.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(str(Config.dataset_path / 'train.csv'))
test_df = pd.read_csv(str(Config.dataset_path / 'test.csv'))

def extract_features(df):
    return df[['Area harvested']]

train_features = extract_features(train_df)
test_features = extract_features(test_df)

train_features.to_csv(str(Config.features_path / 'train_features.csv'), index = None)
test_features.to_csv(str(Config.features_path / 'test_features.csv'), index = None)

train_df.Production.to_csv(str(Config.features_path / 'train_target.csv'), index = None)
test_df.Production.to_csv(str(Config.features_path / 'test_target.csv'), index = None)