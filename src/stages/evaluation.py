import argparse
import joblib
import json
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, log_loss
from typing import Text, Dict
import yaml

from src.report.visualization import plot_confusion_matrix



def evaluate_model(config_path: Text) -> None:
    """Evaluate model.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    model_path = config['model']['path']
    model = joblib.load(model_path)

    test_df = pd.read_csv(config['data']['test_db'])

    target_column=config['train']['target_column']
    y_test = test_df.loc[:, target_column].values
    X_test = test_df.drop(target_column, axis=1).values

    y_pred = model.predict(X_test)
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
    accuracy = accuracy_score(y_test,y_pred)
    loss = log_loss(y_true=y_test, y_pred=y_pred)
    cm = confusion_matrix(y_pred, y_test)
    report = {
        'f1': f1,
        'accuracy': accuracy,
        'loss': loss,
        'cm': cm,
        'actual': y_test,
        'predicted': y_pred
    }

    # save f1 metrics file

    metrics_path = config['reports']['metrics_file']

    json.dump(
        obj={'f1_score': report['f1'],
             'accuracy': report['accuracy'],
             'loss': report['loss']},
        fp=open(metrics_path, 'w')
    )

    # save confusion_matrix.png
    plt = plot_confusion_matrix(cm=report['cm'],
                                target_names=['not_diabetes', 'diabetes'],
                                normalize=False)
    confusion_matrix_png_path = config['reports']['confusion_matrix_image']
    plt.savefig(confusion_matrix_png_path)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)