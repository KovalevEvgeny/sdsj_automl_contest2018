import argparse
import os
import pandas as pd
import pickle
import time
from catboost import CatBoostRegressor, CatBoostClassifier
from fp import load_data

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--prediction-csv', type=argparse.FileType('w'), required=True)
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()

    start_time = time.time()

    # load model
    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'rb') as fin:
        model_config = pickle.load(fin)

    X_test, _, _, line_id = load_data(args.test_csv, datatype='test', cfg=model_config)

    #model = model_config['model']

    if model_config['mode'] == 'regression':
        model = CatBoostRegressor()
        model.load_model(os.path.join(args.model_dir, 'model.catboost'))
        prediction = model.predict(X_test)
    elif model_config['mode'] == 'classification':
        model = CatBoostClassifier()
        model.load_model(os.path.join(args.model_dir, 'model.catboost'))
        prediction = model.predict_proba(X_test)[:, 1]

    result = pd.DataFrame({'line_id': line_id, 'prediction': prediction})
    result.to_csv(args.prediction_csv, index=False)

    print('Prediction time: {}'.format(time.time() - start_time))
