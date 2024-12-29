import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def data_cleansing(dataframe = None, y = None, istrain = True,  train_scaler = None, output=1 ):
    # 1- Processing Missing Data Deleting any columns or rows that have missing values.
    dataframe = dataframe.join(y)
    dataframe = dataframe.dropna()
    # 3- Removing Duplicate Data: 
    dataframe = pd.DataFrame(dataframe)
    dataframe = dataframe.drop_duplicates()

    y = dataframe[dataframe.columns[-output]]
    del dataframe[dataframe.columns[-output]]

    if istrain:
        #scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = StandardScaler()
        #scaler = RobustScaler()
        dataframe1 = scaler.fit_transform(dataframe)
        return dataframe1, y, scaler
    else:
        dataframe1 = train_scaler.transform(dataframe)
        return dataframe1, y
    
    