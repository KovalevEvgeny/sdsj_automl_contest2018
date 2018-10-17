import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

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
        df['number_year_{}'.format(col_name)] = df[col_name].apply(lambda x: x.year)
        df['number_weekday_{}'.format(col_name)] = df[col_name].apply(lambda x: x.weekday())
        df['number_month_{}'.format(col_name)] = df[col_name].apply(lambda x: x.month)
        df['number_day_{}'.format(col_name)] = df[col_name].apply(lambda x: x.day)
        df['number_hour_{}'.format(col_name)] = df[col_name].apply(lambda x: x.hour)
    return df.drop(datetime_columns, axis=1)

# type division
def is_binary(X):
    '''
  X: pd.Series
  return: bool
  '''
    return len(X.dropna().unique()) == 2

def is_categorical(X):
    '''
  X: pd.Series
  return: bool
  '''
    return len(X.dropna().unique()) > 2 and len(X.dropna().unique()) <= 15 # better to compare with global variable 

def get_types(df):
    '''
  df: pd.DataFrame
  return: dictionary, keys - *real* type, values - lists of cols names
  
  '''
    cols_type = {
      'numeric' : [],
      'binary' : [],
      'categorical' : [],
      'other' : []
  }
  
    num_str_cols = [col_name for col_name in df.columns 
                  if col_name.startswith(('number','string'))]
  
    for col in num_str_cols:
        if (col.find('datetime') != -1 and col.find('year') == -1 ):
            cols_type['other'].append(col)            
        elif is_binary(df[col]):
            cols_type['binary'].append(col)
        elif is_categorical(df[col]):
            cols_type['categorical'].append(col)
        else:
            if col.startswith('number'):
                cols_type['numeric'].append(col)
            elif col.startswith('string'):
                print('Warning: maybe text or category (exceeded fixed limit %i, found %i)' % (15, len(df[col])))
                cols_type['categorical'].append(col) # WARN
            else:
                pass        

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

# after get_types
def encoding(df, method='labels', le_cols={}):
    if method == 'labels':
        # train
        if not le_cols:
            for col in df.columns:
                if col.startswith('string'):
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    # values seen by LE
                    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
                    le_cols[col] = (le, le_dict)
        # test
        else:
            for col in le_cols.keys():
                le, le_dict = le_cols[col]
                # handling unseen values
                #for value in df[col].unique():
                #    if value not in le_dict.keys():
                #        le_dict[value] = -1
                diff = list(set(df[col]) - set(le_dict.keys()))
                if not diff:
                    le_dict.update({key:-1 for key in diff})
                #df_categorical[col] = le.transform(df[col].astype(str))
                df[col] = df[col].apply(lambda x: le_dict.get(x))
    elif method == 'one-hot':
        #df_categorical = pd.get_dummies(df_cat.applymap(str))
        print('not yet developed')

    return df, le_cols

def load_data(filename, datatype='train', cfg={}):

    model_config = cfg

    # read dataset
    df = pd.read_csv(filename, low_memory=False)
    line_id = df['line_id'].values
    if datatype == 'train':
        y = df.target
        df = df.drop('target', axis=1)
        if df.memory_usage().sum() > 500 * 1024 * 1024:
            model_config['is_big'] = True
    else:
        y = None       
    print('Dataset read, shape {}'.format(df.shape))
    #print(df.columns)

    # features from datetime
    df = transform_datetime_features(df)
    print('Transform datetime done, shape {}'.format(df.shape))
    #print(df.columns)
    
    # drop irrelevant columns
    if datatype == 'train':
        df = drop_irrelevant(df)
    else:
        df = df[model_config['used_columns']]
    print('Irrelevant columns dropped, shape {}'.format(df.shape))
    #print(df.columns)
    
    if datatype == 'train':
        column_types = get_types(df)
        model_config['column_types'] = column_types
    else:
        column_types = model_config['column_types']
    
    numeric = column_types['numeric'] #float16
    binary_categorical = column_types['binary'] + column_types['categorical'] #uint8
    other = column_types['other']
    
    # missing values
    df.loc[:, numeric] = imputing_missing_values(df.loc[:, numeric], fill_type='num')
    df.loc[:, binary_categorical] = imputing_missing_values(df.loc[:, binary_categorical], fill_type='cat')
    df.loc[:, other] = imputing_missing_values(df.loc[:, other], fill_type='cat')
    print('Missing values imputed')
    #print(df.columns)
    
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
    
    # encoding
    if datatype == 'train':
        df.loc[:, binary_categorical], model_config['le_cols'] = encoding(df.loc[:, binary_categorical], le_cols={})
    else:
        df.loc[:, binary_categorical], _ = encoding(df.loc[:, binary_categorical], le_cols=model_config['le_cols'])
    print('Encoding done')
    #print(df.columns)
    
    
    # downcasting types
    df.loc[:, numeric] = df.loc[:, numeric].apply(pd.to_numeric, downcast='float')
    df.loc[:, binary_categorical] =  df.loc[:, binary_categorical].apply(pd.to_numeric, downcast='unsigned')
    
    #df = pd.concat([df_numeric, df_binary_categorical, df_other], axis=1)
    
    # filtering columns
    if datatype == 'train':
        model_config['used_columns'] = df.columns
    
    print('Used {} columns'.format(len(df.columns)))
    

    return df, y, model_config, line_id





