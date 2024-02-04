from typing import Text
import yaml
import pandas as pd
import argparse
import joblib
from src.train.train import train


def train_model(config_path: Text) -> None:

    
    with open('params.yaml') as config_file:
        config = yaml.safe_load(config_file)

    train_db = pd.read_csv(config['data_split']['train_db'])
    
    estimator_used = config['train']['estimators_used']

    model = train(
        df=train_db,
        target_column=config['featurize']['target_column'],
        estimator_name=estimator_used,
        param_grid=config['train']['estimators'][estimator_used]['param_grid'],
        cv= config['train']['cv']
    )

    model_path = config['model']['path']
    joblib.dump(model, model_path)

    print("train Completato!")



if __name__ == '__main__':
    # se volessimo runnare questo modulo tramite terminale
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config', required=True)
    #permette di parsare la configurazione alla linea di comando
    args = arg_parser.parse_args()

    train_model(config_path=args.config)