from typing import Text
import yaml
import pandas as pd
import argparse

def featurize(config_path: Text) -> None:

    with open('params.yaml') as config_file:
        config = yaml.safe_load(config_file)

    raw_db = pd.read_csv(config['data']['raw_db'])
    diabetes_features = raw_db[[
        'pregnancies', 'glucose', 'diastolic', 'triceps',
        'insulin', 'bmi',
        'target'
    ]]      

    diabetes_features.to_csv(config['data']['featurize_db'])

    print("featurize Completato!")


if __name__ == '__main__':
    # se volessimo runnare questo modulo tramite terminale
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config', required=True)
    #permette di parsare la configurazione alla linea di comando
    args = arg_parser.parse_args()

    featurize(config_path=args.config)
