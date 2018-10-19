import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

start_date = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d')

# datetime
def parse_dt(x):
    if not isinstance(x, str):
        return start_date
    elif len(x) == len('2010-01-01'):
        return datetime.datetime.strptime(x, '%Y-%m-%d')
    elif len(x) == len('2010-01-01 10:10:10'):
        return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    else:
        return start_date

def transform_datetime_features(df):
    datetime_columns = [
        col_name
        for col_name in df.columns
        if col_name.startswith('datetime')
    ]
    for col_name in datetime_columns:
        df[col_name] = df[col_name].apply(lambda x: parse_dt(x))
        df['number_datetime_year_{}'.format(col_name)] = df[col_name].apply(lambda x: x.year)
        df['number_datetime_weekday_{}'.format(col_name)] = df[col_name].apply(lambda x: x.weekday())
        df['number_datetime_month_{}'.format(col_name)] = df[col_name].apply(lambda x: x.month)
        df['number_datetime_day_{}'.format(col_name)] = df[col_name].apply(lambda x: x.day)
        df['number_datetime_hour_{}'.format(col_name)] = df[col_name].apply(lambda x: x.hour)
        df['number_datetime_minute_{}'.format(col_name)] = df[col_name].apply(lambda x: x.minute)
        df['number_datetime_second_{}'.format(col_name)] = df[col_name].apply(lambda x: x.second)
    return df.drop(datetime_columns, axis=1)

# type division
def is_categorical(X):
    '''
  X: pd.Series
  return: bool
  '''
    return len(X.dropna().unique()) <= 15 # better to compare with global variable 


def get_types(df):
    '''
  df: pd.DataFrame
  return: dictionary, keys - *real* type, values - lists of cols names
  
  '''
    cols_type = {
      'numeric' : [],
      'categorical_number' : [],
      'categorical_string' : [],
      'must_be_empty': []
  }
  
    for col in df.columns:
        if col.startswith('string'):
            cols_type['categorical_string'].append(col)
        elif col.startswith('number'):
            if is_categorical(df[col]) or col.startswith('number_datetime'):
                cols_type['categorical_number'].append(col)
            else:
                cols_type['numeric'].append(col)
        else:
            # I cannot get into this "else" I assume
            cols_type['must_be_empty'].append(col)
    
    return cols_type    


def drop_irrelevant(df):
  # constant
  # NA >> not NA
    irrelevant_columns = [
      col_name
      for col_name in df.columns
      if df[col_name].dropna().nunique() == 0
      or df[col_name].dropna().nunique() == 1
      or df[col_name].isnull().sum() / len(df) >= 0.98
      ]
    return df.drop(irrelevant_columns, axis=1)

# after get_types
def imputing_missing_values(df, fill_type, cat_method='mode', num_method='mean'):
    if fill_type == 'cat':
        if cat_method == 'mode':
            return df.fillna(value=df.mode().iloc[0])
        elif cat_method == 'nan':
            print('not yet developed')
    elif fill_type == 'num':
        if num_method == 'mean':
            return df.fillna(value=df.mean())
        elif num_method == 'median':
            return df.fillna(value=df.median())
        elif num_method == 'other':
            print('not yet developed')

    #other_columns = column_types['other']
    #df_other = df.loc[:, other_columns]
# after get_types
def scaling(df, scalers):
    if len(scalers) == 1: #then it is train
        if scalers['method'] == 'minmax':
            sc = MinMaxScaler()
            df = sc.fit_transform(df)
            scalers['sc'] = sc
        elif scalers['method'] == 'std':
            sc = StandardScaler()
            df = sc.fit_transform(df)
            scalers['sc'] = sc
    else:
        df = scalers['sc'].transform(df)       
    
    return df, scalers


def load_data(filename, datatype='train', cfg={}):

    model_config = cfg

    # read dataset
    df = pd.read_csv(filename)
    line_id = df['line_id'].values
    if datatype == 'train':
        # subsampling for huge df
        if df.memory_usage().sum() > 500 * 1024 * 1024:
            df = df.sample(frac=0.25, random_state=13)
        y = df.target
        df = df.drop('target', axis=1)
    else:
        y = None       
    print('Dataset read, shape {}'.format(df.shape))
    #print(df.columns)

    
    # features from datetime
    df = transform_datetime_features(df)
    print('Transform datetime done, shape {}'.format(df.shape))
    #print(df.columns)
    
    
    # downcasting types
    print(df.info(verbose=False, memory_usage='deep'))
    print('Downcasting started')
    df.loc[:, df.dtypes == np.int64] = df.loc[:, df.dtypes == np.int64].apply(pd.to_numeric, downcast='unsigned')
    df.loc[:, df.dtypes == np.float64] = df.loc[:, df.dtypes == np.float64].apply(pd.to_numeric, downcast='float')
    
    obj_columns = df.select_dtypes(include=['object'])
    for col in obj_columns:
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:
            df.loc[:, col] = df[col].astype('category')
        else:
            df.loc[:, col] = df[col]
    print(df.info(verbose=False, memory_usage='deep'))
    print('Downcasting finished')
    
    
    
    # drop irrelevant columns
    if datatype == 'train':
        df = drop_irrelevant(df)
        # drop id_... and line_id
        df = df[df.columns.drop(list(df.filter(regex='id')))]
    else:
        df = df[model_config['used_columns']]
    print('Irrelevant columns dropped, shape {}'.format(df.shape))
    #print(df.columns)
    
    if datatype == 'train':
        column_types = get_types(df)
        model_config['column_types'] = column_types
    else:
        column_types = model_config['column_types']
    
    numeric = column_types['numeric']
    categorical_string = column_types['categorical_string']
    categorical_number = column_types['categorical_number']
    
    categorical_string_indices = np.nonzero(df.columns.isin(categorical_string))[0]
    model_config['cat_features'] = categorical_string_indices
    
    # missing values
    df.loc[:, numeric] = imputing_missing_values(df.loc[:, numeric], fill_type='num')
    df.loc[:, categorical_string + categorical_number] = imputing_missing_values(df.loc[:, categorical_string + categorical_number], fill_type='cat')
    print('Missing values imputed')
    #print(df.columns)
    
    '''
    # scaling
    if numeric:
        if datatype == 'train':
            df.loc[:, numeric], model_config['scalers'] = scaling(df.loc[:, numeric], scalers={'method':'minmax'})
        else:
            df.loc[:, numeric], _ = scaling(df.loc[:, numeric], scalers=model_config['scalers'])
        print('Scaling done')
    else:
        print('Nothing to scale')
    #print(df.columns)
    '''

    
    #df = pd.concat([df_numeric, df_binary_categorical, df_other], axis=1)
    
    # filtering columns
    if datatype == 'train':
        model_config['used_columns'] = df.columns
    
    print('Used {} columns'.format(len(df.columns)))
    

    return df, y, model_config, line_id





