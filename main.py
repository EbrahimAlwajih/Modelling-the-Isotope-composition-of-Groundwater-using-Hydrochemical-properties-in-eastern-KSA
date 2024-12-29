from datetime import datetime
import os
import uuid
import random
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error, median_absolute_error,
                             r2_score, explained_variance_score, mean_absolute_percentage_error)
from sklearn.ensemble import (StackingRegressor, ExtraTreesRegressor, RandomForestRegressor, 
                              AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from catboost import CatBoostRegressor

from CleaningData import data_cleansing

random.seed(42)
np.random.seed(42)

dataset_name = "δ18O"  
current_path = os.getcwd()
# Get the machine ID
machine_id = uuid.getnode()
current_date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
results_folder = os.path.join(current_path, f"results_{dataset_name}_{machine_id}_{current_date}")
os.makedirs(results_folder, exist_ok=True)

url = '.\\'+dataset_name+'.csv'
dataframe = pd.read_csv(url, header=0)
describe = dataframe.describe(include='all')
print(describe)
random_state = 10  

classes = dataframe[dataframe.columns[-1]]
features = dataframe[dataframe.columns[:-1]]
X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.20, random_state=42) 
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
        'ET': ExtraTreesRegressor(random_state=random_state),
        'AdaBt': AdaBoostRegressor(random_state=random_state),
        'GRB': GradientBoostingRegressor(random_state=random_state),
        'Bag' : BaggingRegressor(random_state=random_state), 
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

def get_best_model(models, results_folder):
        trained_models = {}
        best_models_params = []
        # DataFrame to store results
        results = []
        for name, model in models.items():
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                for metric_name, metric_function in evaluation_metrics.items():
                        if metric_name == 'Adjusted R2':
                                scores[metric_name][name]  = metric_function(y_test, y_pred, X_test)
                        else:
                                scores[metric_name][name]  = metric_function(y_test, y_pred)   
                trained_models[name] = model
                joblib_file_path = os.path.join(results_folder, 'Models' , name + "_model.joblib")
                os.makedirs(os.path.join(results_folder, 'Models'), exist_ok=True)
                joblib.dump(model, joblib_file_path)
                print('{} for Train set {}: {}'.format(name,  refit_score, model.score(X_train, y_train)))      
                print('{} for Test set {}: {}'.format(name,  refit_score, model.score(X_test, y_test)))
                #plotGraph(y_test, y_pred, "test")
        
        # Convert results to DataFrame and save to CSV
        results_folder = os.path.join(current_path, f"results_{dataset_name}_{machine_id}_{current_date}")
        os.makedirs(results_folder, exist_ok=True)
        csv_filename = "results_metrics.csv"
        csv_file_path = os.path.join(results_folder, csv_filename)
        results_df = pd.DataFrame(scores)
        df_sorted = results_df.sort_values(by=['R2'], ascending=[False])
        df_sorted.to_csv(csv_file_path, index=True)
        return trained_models, scores


def plot_models_comparison(top_models):
        for metric_name, metric_func in evaluation_metrics.items():
                metric_values = []  # To store metric values for each model
                for index, model_config in enumerate(top_models[:5], start=0):
                        stacking_regressor = top_models[index-1]['Trained_Model']
                        y_pred = stacking_regressor.predict(X_test)
                        if metric_name == 'Adjusted R2':
                                metric_value = metric_func(y_test, y_pred, X_test)
                        else:
                                metric_value = metric_func(y_test, y_pred)
                        metric_values.append(metric_value)

                        if metric_name == 'R2' :
                                # Plot Predicted vs. Actual values
                                plt.figure(figsize=(10, 6))
                                plt.scatter(y_test, y_pred, alpha=0.5, label='Predictions')
                                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal')
                                plt.xlabel('Actual Values')
                                plt.ylabel('Predicted Values')
                                plt.title(f'Predicted vs. Actual Values - Model {index}')
                                plt.legend()
                                # Save the plot
                                pred_vs_actual_plots_directory = os.path.join(results_folder, "predicted_vs_actual_plots")
                                os.makedirs(pred_vs_actual_plots_directory, exist_ok=True)
                                plot_filename = os.path.join(pred_vs_actual_plots_directory, f'Predicted_vs_Actual_Model_{index}.png')
                                plt.savefig(plot_filename)
                                plt.close()  
        

                # Plot a bar chart for this metric across all top models
                plt.figure(figsize=(10, 6))
                plt.bar(range(1, len(top_models[:5]) + 1), metric_values, color='skyblue', alpha=0.7)
                plt.xticks(range(1, len(top_models[:5]) + 1), [f'Model {i}' for i in range(1, len(top_models[:5]) + 1)])
                plt.ylabel('Value')
                plt.title(f'{metric_name} Comparison Across Top 5 Models')
                # Save the plot
                metrics_comparison_directory = os.path.join(results_folder, "metrics_comparison_charts") 
                os.makedirs(metrics_comparison_directory, exist_ok=True)
                plot_filename = os.path.join(metrics_comparison_directory, f'{metric_name}_Comparison.png')
                plt.savefig(plot_filename)
                plt.close()


def train_stacking_models(best_models):
        model_combinations = []
        for r in range(len(best_models) + 1):
                model_combinations.extend(list(combinations(best_models.items(), r)))
        top_models = []
        idx = -1
        trained_models = []

        for combo in model_combinations:
                for meta_model_name, meta_model in combo:
                        idx = idx + 1
                        base_learners = [(name, model) for name, model in combo if name != meta_model_name]
                        base_learner_names = ', '.join([name for name, _ in base_learners])

                        if base_learner_names:
                                stacking_regressor = StackingRegressor(estimators=base_learners, final_estimator=meta_model)
                                model = stacking_regressor.fit(X_train,y_train)
                                y_pred = model.predict(X_test)
                                for metric_name, metric_function in evaluation_metrics.items():
                                        if metric_name == 'Adjusted R2':
                                                scores[metric_name][model]  = metric_function(y_test, y_pred, X_test)
                                        else:
                                                scores[metric_name][model]  = metric_function(y_test, y_pred)   
                                model_config = {
                                        'Meta Model': meta_model_name,
                                        'Base Models': base_learner_names,
                                        'R2' : scores['R2'][model],
                                        'Trained_Model' : model,
                                        # 'Scores': scores[metric_name][model],
                                        'Scores': {outer_key: (lambda d: d.get(model, None))(inner_dict) for outer_key, inner_dict in scores.items()}
                                }
                                top_models.append(model_config)
                                top_models = sorted(top_models, key=lambda x: x['R2'], reverse=True)[:5]  # Keep only top 5


        for i in range(len(top_models)):
                Trained_Model = top_models[i]['Trained_Model']
                joblib_file_path = os.path.join(results_folder, 'Models' , "Model" + str(i) + "_Stacked_model.joblib")
                os.makedirs(os.path.join(results_folder, 'Models'), exist_ok=True)
                joblib.dump(top_models[i]['Trained_Model'], joblib_file_path)
        
                csv_file_path = os.path.join(results_folder, 'Results' , "Model" + str(i) + "_Stacked_model.csv")
                os.makedirs(os.path.join(results_folder, 'Results'), exist_ok=True)
                results_df = pd.DataFrame(top_models[i])
                results_df.to_csv(csv_file_path)

        top_models_df = pd.DataFrame(top_models)
        csv_filename = "top_5_stacking_models.csv"
        csv_file_path = os.path.join(results_folder, csv_filename)
        top_models_df.to_csv(csv_file_path, index=False)
        print("Top 5 stacking models' details have been saved to 'top_5_stacking_models.csv'")
        plot_models_comparison(top_models)


scores = {metric: {model_name: [] for model_name,_ in models.items()} for metric in evaluation_metrics.keys()}    
trained_models, scores = get_best_model(models, results_folder)
  
# Create bar charts to visualize the evaluation metrics for each model
fig, axes = plt.subplots(len(evaluation_metrics), 1, figsize=(8, 12))
plt.subplots_adjust(hspace=0.5)
for i, (metric_name, metric_scores) in enumerate(scores.items()):     
        ax = axes[i]
        print(metric_scores.keys())
        print(metric_scores.values())
        ax.bar(metric_scores.keys(), metric_scores.values())
        ax.scatter(metric_scores.keys(), metric_scores.values())
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} for Linear Regression Models')
plt.tight_layout()
metrics_comparison_directory = os.path.join(results_folder, "Basic_models_charts") 
os.makedirs(metrics_comparison_directory, exist_ok=True)
plot_filename = os.path.join(metrics_comparison_directory, 'Basic_models.png')
plt.savefig(plot_filename)
plt.close()

train_stacking_models(trained_models)