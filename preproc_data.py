import pandas as pd


class Parse():
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.cols = self.data.columns
        self.l, self.d = self.data.shape

    def get_new_cols(self):
        col_typ = {col : typ for col, typ in zip(self.cols, self.data.dtypes)}
        
        for i, c in enumerate(self.data.select_dtypes(['number']).columns.values.tolist()):
            col_typ[c] = 'number_' + str(i)
       
        for i, c in enumerate(self.data.select_dtypes(['O', 'category']).columns.values.tolist()):
            try:
                self.data[c] = pd.to_datetime(self.data[c], format='%Y-%m-%d', errors='raise')
            except ValueError:
                col_typ[c] = 'string_' + str(i)
        
        for i, c in enumerate(self.data.select_dtypes(['datetime']).columns.values.tolist()):
            col_typ[c] = 'datetime_' + str(i)
            
        self.col_typ = col_typ

    def rename_cols(self):
        self.data.rename_axis(self.col_typ, axis='columns', inplace=True)
        
    def save_csv(self, name):
        self.data.to_csv(name)
        
        
        

    