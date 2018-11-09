import numpy as np
import pandas as pd

# datetime

def transform_datetime_features(df, datatype):
    datetime_columns = [
        col_name
        for col_name in df.columns
        if col_name.startswith('datetime')
    ]
    for col_name in datetime_columns:
        if datatype == 'train':
            df[col_name] = pd.to_datetime(df[col_name])
        df['number_datetime_year_{}'.format(col_name)] = df[col_name].dt.year
        df['number_datetime_weekday_{}'.format(col_name)] = df[col_name].dt.weekday
        df['number_datetime_month_{}'.format(col_name)] = df[col_name].dt.month
        df['number_datetime_day_{}'.format(col_name)] = df[col_name].dt.day
        df['number_datetime_hour_{}'.format(col_name)] = df[col_name].dt.hour
        df['number_datetime_minute_{}'.format(col_name)] = df[col_name].dt.minute
        df['number_datetime_second_{}'.format(col_name)] = df[col_name].dt.second
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


def load_data(filename, datatype='train', cfg={}):

    model_config = cfg

    # read dataset
    if datatype == 'train':
        df = pd.read_csv(filename, low_memory=False)
    else:
        df = pd.read_csv(filename, usecols=model_config['usecols'], dtype=model_config['dtype'],
                         parse_dates=model_config['parse_dates'])
    line_id = df['line_id'].values
    if datatype == 'train':
        # subsampling for huge df
        if df.memory_usage().sum() > 500 * 1024 * 1024:
            df = df.sample(frac=0.25, random_state=13)
            model_config['is_big'] = True
        else:
            model_config['is_big'] = False
        y = df.target
        df = df.drop('target', axis=1)
    else:
        y = None
    print('Dataset read, shape {}'.format(df.shape))
    #print(df.columns)
    
    if datatype == 'train':
        model_config['parse_dates'] = [
            col_name
            for col_name in df.columns
            if col_name.startswith('datetime')
        ]
        model_config['usecols'] = [
            col_name
            for col_name in df.columns
            if (col_name.startswith('datetime') or col_name.startswith('id'))
        ]
        model_config['usecols'].append('line_id')
    
    # features from datetime
    df = transform_datetime_features(df, datatype)
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
    
    # filtering columns
    if datatype == 'train':
        model_config['used_columns'] = df.columns
    
    print('Used {} columns'.format(len(df.columns)))
    
    if datatype == 'train':
        model_config['dtype'] = {}
        for col_name in df.columns:
            if not col_name.startswith('number_datetime'):
                model_config['usecols'].append(col_name)
                model_config['dtype'][col_name] = df[col_name].dtype
    

    return df, y, model_config, line_id





