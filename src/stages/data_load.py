from typing import Text
import yaml
import pandas as pd

def data_load(config_path: Text) -> None:

    with open('params.yaml') as config_file:
        config = yaml.safe_load(config_file)

    diabetes = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Diabetes.csv')
    diabetes.rename(columns={'diabetes':'target'}, inplace=True)
    diabetes.to_csv(config['data']['raw_db'])
    