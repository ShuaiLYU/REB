

import numpy as np 

import os
import csv

import pandas as pd



# 保留 小数

def  keep_decimal_places(num,places):
    if isinstance(num,float):
        return round(num,places)
    else:
        return num



# 转置


def transpose(data):
    return data.T




from typing import List
from typing import Union



def df_from_dicts(data:List[dict]):
    return pd.DataFrame(data)

def df_to_dicts(data):
    return  [ row for idx, row in data.iterrows()]

class CsvReader(object):
    def __init__(self,csv_path):
        self.csv_path=csv_path
        self.header,self.rows=self._read_csv(csv_path)
        
    def _read_csv(self,csv_path):
        with open(csv_path, newline='') as csvfile:
            spamreader = [  row for row  in csv.reader(csvfile, delimiter=',', quotechar='|') ]
            header=spamreader[0]
            rows=[]
            for row_val in spamreader[1:]:
                rows.append({ key:val for key,val in zip(header,row_val)})
            return header,rows
    def get_last_row(self):
        return self.rows[-1]

    
class CsvLogger(object):
    
    
    def __init__(self,root,csv_name):
        suffix=".csv"
        self.root=root
        self.csv_name=csv_name if csv_name.endswith(suffix) else csv_name+suffix
        self.csv_path=os.path.join(self.root,self.csv_name)
        
        if not  os.path.exists(self.root):
            os.makedirs(self.root)
        self.header=None

        if os.path.exists(os.path.join(self.root,self.csv_name)):
            self.header,_=self._read_csv(self.csv_path)
            print("found a existing csv file and load the header: {}...".format(self.header))
            self.flag=0
        else:
            self.flag=1
        
    def exists(self):
        return os.path.exists(os.path.join(self.root,self.csv_name))
    
    def if_new(self):
        return self.flag
    
    
    def set_header(self,header:list):
        
        assert(self.header==None)
        self.header=header    
        self.append_one_row({ k:k for k in self.header})
        return self
    
    def get_rows(self):
        _,rows=self._read_csv(self.csv_path)
        return rows
    
    # def _read_csv(self,csv_path):
    #     with open(csv_path, newline='') as csvfile:
    #         spamreader = [  row for row  in csv.reader(csvfile, delimiter=',', quotechar='|') ]
    #         header=spamreader[0]
    #         rows=[]
    #         for row_val in spamreader[1:]:
    #             rows.append({ key:val for key,val in zip(header,row_val)})
    #         return header,rows

    def _read_csv(self,csv_path):
        import pandas as df
        df_data = df.read_csv(csv_path)
        header= df_data.columns.to_list()
        rows=[ row.to_dict() for  idx, row in df_data.iterrows()]
        return header,rows

# def _save_csv_file(self,df):
        
    def append_one_row(self,row:dict,strict=True):
        if self.header is None:
            self.set_header(row.keys())
        
        if strict:
            assert(len(row)==len(self.header))
            assert( all([ (k in self.header) for k,v in row.items()])),row
            row= [  row[k] for k in  self.header]
            with open(self.csv_path, 'a+', newline='') as csvfile:
                csvw = csv.writer(csvfile)
                csvw.writerow(row)
        else:
            raise NotImplementedError
            #         try:
            #         except PermissionError:  
            #         return


################pandas

def get_max_row_by(df,key):
    ind = df[key].idxmax()
    # row = df.iloc[ind,:]
    return ind