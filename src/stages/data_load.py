from typing import Text
import yaml
import pandas as pd
import argparse

def data_load(config_path: Text) -> None:

    with open('params.yaml') as config_file:
        config = yaml.safe_load(config_file)

    diabetes = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Diabetes.csv')
    diabetes.rename(columns={'diabetes':'target'}, inplace=True)
    diabetes.to_csv(config['data_load']['raw_db'], index=False)

    print('data_load Completato!')
    

if __name__ == '__main__':
    # se volessimo runnare questo modulo tramite terminale
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config', required=True)
    #permette di parsare la configurazione alla linea di comando
    args = arg_parser.parse_args()

    data_load(config_path=args.config)
