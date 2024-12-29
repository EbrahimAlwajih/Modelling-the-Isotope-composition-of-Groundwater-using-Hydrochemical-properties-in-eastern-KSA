from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from joblib import dump, load
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from CleaningData import data_cleansing
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from CompareModels import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error
from catboost import CatBoostRegressor
import os,  joblib, uuid
from scipy.stats import pearsonr


dataset_name = "δ18O"  
current_path = os.getcwd()
# Get the machine ID
machine_id = uuid.getnode()

url = '.\\'+dataset_name+'.csv'
dataframe = pd.read_csv(url, header=0)
describe = dataframe.describe(include='all')
print(describe)
random_state=42
classes = dataframe[dataframe.columns[-1]]
features = dataframe[dataframe.columns[:-1]]
X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.20, random_state=42) # random_state=0
n_samples, n_features = X_test.shape
##########################################

X_train, y_train, scaler = data_cleansing(dataframe = X_train, y = y_train)
X_test, y_test = data_cleansing(dataframe = X_test, y = y_test, istrain = False, train_scaler = scaler)

##################################################################
# #1- Filter Methods:
# 1.1 Correlation-based Feature Selection
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(score_func=f_regression, k= 3)  # δ18O 8 0.954842068 20%,  δ18O_SecondData 8 0.956533881
X_train_selected = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)
# Get the boolean mask of the selected features
mask = selector.get_support()

# Get the names of the selected features
selected_features = features.columns[mask]

# Create a new DataFrame with the selected features
X_new_df = pd.DataFrame(X_train_selected, columns=selected_features)

# X_new_df now is a DataFrame with the selected features and their names
print(X_new_df.head())
X_train = X_train_selected


# Calculate metrics
evaluation_metrics  = {
        'MAE': mean_absolute_error,
        'MSE': mean_squared_error,
        'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score,
        # 'Adjusted R2' : lambda y_true, y_pred, X: adjusted_r2_score(y_true, y_pred, X),
        'MAPE': mean_absolute_percentage_error,
        'MedAE': median_absolute_error,
        'EV': explained_variance_score,
        'NSE': lambda y_true, y_pred: 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2),
        'pearson_corr': lambda x, y: pearsonr(x, y)[0]  # [0] to get the correlation coefficient
}

refit_score = 'neg_mean_squared_error'


models = {
        'SVR': svm.SVR(),
        'KNN':  KNeighborsRegressor(),
        'RF': RandomForestRegressor(random_state=random_state),
        'ET': ExtraTreesRegressor(),
        'AdaBt': AdaBoostRegressor(),
        'GRB': GradientBoostingRegressor(),
        'Bag' : BaggingRegressor(), 
        'CAT': CatBoostRegressor(verbose=0, random_seed=random_state),
}



def plotGraph(y_test,y_pred,regressorName):
    if max(y_test) >= max(y_pred):
        my_range = int(max(y_test))
    else:
        my_range = int(max(y_pred))
    plt.scatter(range(len(y_test)), y_test, color='blue')
    plt.scatter(range(len(y_pred)), y_pred, color='red')
    plt.title(regressorName)
    plt.show()
    return

def save_predected_values(scores):

        results_folder = os.path.join('J:\\My Drive\\Research\\AI Reaserch Team - KFUPM\\code_v3\\Best Models\\V2\\results_' + dataset_name +'_(3 att_v2)')
        directory_path = os.path.join(results_folder, 'Models')  
        all_files_and_directories = os.listdir(directory_path)
        

        # Filter out directories, keep only files
        trained_models = [f for f in all_files_and_directories if os.path.isfile(os.path.join(directory_path, f))]

        # print(trained_models)

        output_df = pd.DataFrame()
        output_df['y_test'] = y_test


        for model_name in trained_models:
                full_path = os.path.join(directory_path, model_name)
                model = joblib.load(full_path)
                model_name=model_name.split("_")[0]

                y_pred = model.predict(X_test)
                output_df[model_name] = y_pred
                for metric_name, metric_function in evaluation_metrics.items():
                        if metric_name == 'Adjusted R2':
                                scores[metric_name][model_name]  = metric_function(y_test, y_pred, X_test)
                        else:
                                scores[metric_name][model_name]  = metric_function(y_test, y_pred)   
                print('{} for Test set {}: {}'.format(model_name,  refit_score, model.score(X_test, y_test)))
                #plotGraph(y_test, y_pred, "test")
        
        os.makedirs(results_folder, exist_ok=True)
        csv_filename = "All_results.csv"
        y_pred_df_filename='y_pred.csv'
        csv_file_path = os.path.join(results_folder, csv_filename)
        y_pred_file_path = os.path.join(results_folder, y_pred_df_filename)

        results_df = pd.DataFrame(scores)
        df_sorted = results_df.sort_values(by=['R2'], ascending=[False])
        df_sorted.to_csv(csv_file_path, index=True)
        output_df.to_csv(y_pred_file_path, index=True)
        return scores

 
scores = {metric: {model_name: [] for model_name,_ in models.items()} for metric in evaluation_metrics.keys()}    
scores = save_predected_values(scores)
