import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config
import gdown

np.random.seed(Config.random_seed)

Config.original_dataset_file_path.parent.mkdir(parents=True, exist_ok=True)
Config.dataset_path.mkdir(parents=True, exist_ok=True)

gdown.download(
    'https://drive.google.com/uc?id=1q7c0UqDGMzYI67kzbpq9xAfBXHMkR_O_',
    str(Config.original_dataset_file_path)
)

df = pd.read_csv(str(Config.original_dataset_file_path), encoding = 'latin1')

df_train, df_test = train_test_split(df, test_size = 0.2, random_state = Config.random_seed)

df_train.to_csv(str(Config.dataset_path / 'train.csv'), index = None)
df_test.to_csv(str(Config.dataset_path / 'test.csv'), index = None)