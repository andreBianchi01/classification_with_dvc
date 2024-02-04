from typing import Text
import yaml
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split


def split_dataset(config_path: Text) -> None:

    with open('params.yaml') as config_file:
        config = yaml.safe_load(config_file)

    
    features_db = pd.read_csv(config['featurize']['featurize_db'])

    y = features_db['target']
    X = features_db.drop(['target'],axis=1)

    train_db, test_db, y_train, y_test = train_test_split(
        y,
        X, 
        test_size=config['data_split']['test_size'], 
        random_state=config['base']['random_state']
    )

    train_db.to_csv(config['data_split']['train_db'])
    test_db.to_csv(config['data_split']['test_db'])

    print("data_split completato!")


if __name__ == '__main__':
    # se volessimo runnare questo modulo tramite terminale
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config', required=True)
    #permette di parsare la configurazione alla linea di comando
    args = arg_parser.parse_args()

    split_dataset(config_path=args.config)