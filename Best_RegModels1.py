from math import sqrt
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from joblib import dump, load
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, LeaveOneOut
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, HuberRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import KNeighborsRegressor
from CleaningData import data_cleansing
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn import svm
from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from CompareModels1 import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error
from catboost import CatBoostRegressor
import os,  joblib, uuid
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


dataset_name = "δ18O_v2"  
current_path = os.getcwd()
# Get the machine ID
machine_id = uuid.getnode()
current_date = datetime.now().strftime('%Y-%m-%d-%H-%M')
results_folder = os.path.join(current_path, f"results_{dataset_name}_{machine_id}_{current_date}")
os.makedirs(results_folder, exist_ok=True)

url = '.\\δ18O_v2.csv'
dataframe = pd.read_csv(url, header=0)

classes = dataframe[dataframe.columns[-1]]
features = dataframe[dataframe.columns[:-1]]
X_train, X_test, y_train, y_test = train_test_split(features,classes,test_size=0.20, random_state=0) # random_state=0

X_train, y_train, scaler = data_cleansing(dataframe = X_train, y = y_train)
X_test, y_test = data_cleansing(dataframe = X_test, y = y_test, istrain = False, train_scaler = scaler)
cv = 10
 
###################################################################
# Feature Reduction:
# 1- PCA :
pca = PCA(n_components=11) # 11 10
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Define a dictionary to store evaluation metric scores for each model

# The most-used error estimation parameters are: root mean square error (RMSE), mean square error (MSE), mean absolute error 
# (MAE), mean square error on prediction (RMSEP), cross correlation coefficient (R), and standard deviation (SD) [46,47,48,49]. 
# 46 -Shcherbakov, M.V.; Brebels, A.; Shcherbakova, N.L.; Tyukov, A.P.; Janovsky, T.A.; Kamaev, V.A. A Survey of Forecast Error Measures.
#  World Appl. Sci. J. 2013, 24, 171–176. [Google Scholar] [CrossRef] 
# 47- Syntetos, A.A.; Boylan, J.E. The Accuracy of Intermittent Demand Estimates. Int. J. Forecast. 2004, 21, 303–314. [Google Scholar] [CrossRef] 
# 48- Mishra, P.; Passos, D. A Synergistic Use of Chemometrics and Deep Learning Improved the Predictive Performance of near-Infrared Spectroscopy 
# Models for Dry Matter Prediction in Mango Fruit. Chemom. Intell. Lab. Syst. 2021, 212, 104287. [Google Scholar] [CrossRef] 
# 49- Panarese, A.; Bruno, D.; Colonna, G.; Diomede, P.; Laricchiuta, A.; Longo, S.; Capitelli, M. A Monte Carlo Model for determination of 
# binary diffusion coefficients in gases. J. Comput. Phys. 2011, 230, 5716–5721. [Google Scholar] [CrossRef]

scoring = {'RMSE': 'neg_root_mean_squared_error',
           'MAE' : 'neg_mean_absolute_error',
           'MedAE': 'neg_median_absolute_error',
           'MSE' : 'neg_mean_squared_error',
           'r2'  : 'r2',
           'EV'  : 'explained_variance'}
from scipy.stats import pearsonr

# Calculate metrics
evaluation_metrics  = {
        'MAE': mean_absolute_error,
        'MSE': mean_squared_error,
        'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score,
        #'Adjusted R2' : lambda y_true, y_pred, X: adjusted_r2_score(y_true, y_pred, X),
        'MAPE': mean_absolute_percentage_error,
        'MedAE': median_absolute_error,
        'EV': explained_variance_score,
        'NSE': lambda y_true, y_pred: 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2),
        'pearson_corr': lambda x, y: pearsonr(x, y)[0]  # [0] to get the correlation coefficient
}

refit_score = 'neg_mean_squared_error'

def adjusted_r2_score(y_true, y_pred, X):
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    n = X.shape[0]  # Number of observations
    p = X.shape[1]  # Number of features
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2

def Pipeline_models():
        # Linear Regression Pipeline
        pipe_lr = Pipeline([('LR', LinearRegression(fit_intercept=True,n_jobs=4))])
        # Decision Tree Pipeline
        pipe_dt = Pipeline([('DT',DecisionTreeRegressor())])#criterion='absolute_error', min_samples_leaf=2, min_samples_split=2, splitter='best',random_state=42))])
        # Random Forest Pipeline
        pipe_rf = Pipeline([('RF',RandomForestRegressor())])
        # KNeighbors Pipeline
        pipe_knn = Pipeline([('KNN', KNeighborsRegressor(algorithm='auto', metric='minkowski', n_neighbors=2, p=1, weights='distance'))])
        # SVM Pipeline
        pipe_svm = Pipeline([('SVR', svm.SVR())])
        # XGB Pipeline
        pipe_xgb = Pipeline([('XGB', XGBRegressor())])
        # ElasticNet Pipeline     
        pipe_elasticnet = Pipeline([('elasticnet', ElasticNet())])
        # CatBoost Pipeline  
        pipe_cat = Pipeline([('CAT', CatBoostRegressor(verbose=0))])
        # Lasso Regression Pipeline
        pipe_ls = Pipeline([('LS', Lasso())])
        # Ridge Regression Pipeline
        pipe_rdg = Pipeline([('RDG', Ridge())])
        # Huber Regression Pipeline
        pipe_hbr = Pipeline([('HBR', HuberRegressor())])       
        from sklearn.neural_network import MLPRegressor
        # neural_network Regression Pipeline
        pipe_nn = Pipeline([('NN', MLPRegressor())])
        
        grids = [pipe_lr, pipe_dt, pipe_rf, pipe_knn, pipe_svm, pipe_xgb, 
                 pipe_elasticnet, pipe_cat, pipe_ls, pipe_rdg, pipe_hbr, pipe_nn]

        grid_dicts = { 
                0: 'Linear Regression',            1: 'Decision Trees', 
                2: 'Random Forest',                3: 'K-Nearest Neighbors',
                4: 'Support Vector Machines',      5: 'XGBoost',
                6: 'Elastic Net',                  7: 'CAT', 
                8: 'Lasso',                        9:  'Ridge', 
                10: 'Huber',                       11: 'NN'
                }
        grid_dicts = { 
                0: 'LR',            1: 'DT', 
                2: 'RF',            3: 'KNN',
                4: 'SVR',           5: 'XGBoost',
                6: 'Elastic Net',   7: 'CAT', 
                8: 'Lasso',         9:  'Ridge', 
                10: 'Huber',        11: 'NN'
                }

        return grids, grid_dicts

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

def get_best_model(grids, grid_dicts):
        best_models = {}
        for pipe, grid_dict in zip (grids, grid_dicts):
                print(pipe.get_params().keys())
                pipe.fit(X_train,y_train)
                best_models[grid_dicts[grid_dict]] = pipe
                joblib_file_path = os.path.join(results_folder, 'Models' , grid_dicts[grid_dict] + "_model.joblib")
                os.makedirs(os.path.join(results_folder, 'Models'), exist_ok=True)
                joblib.dump(pipe, joblib_file_path)   

                y_pred = pipe.predict(X_test)          
                for metric_name, metric_function in evaluation_metrics.items():
                        if metric_name == 'Adjusted R2':
                                scores[metric_name][grid_dicts[grid_dict]]  = metric_function(y_test, y_pred, X_test)
                        else:
                                scores[metric_name][grid_dicts[grid_dict]]  = metric_function(y_test, y_pred)         
                print('{} Test {}: {}'.format(grid_dicts[grid_dict],  'pipe.score', pipe.score(X_test, y_test)))
                print('{} R2: {}'.format(grid_dicts[grid_dict], scores['R2']))
        return best_models, scores

# Function to evaluate stacking ensemble
def evaluate_stacking_ensemble(estimators, final_estimator, X, y, cv):
    stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=final_estimator)
    cv_results = cross_val_score(stacking_regressor, X, y, cv=cv, scoring='r2')
    Trained_Model = stacking_regressor.fit(X,y)
    return cv_results, Trained_Model

def plot_models_comparison(top_models):
# Iterate over each metric to plot and save its comparison across all top models
        for metric_name, metric_func in evaluation_metrics.items():
                metric_values = []  # To store metric values for each model
                for index, model_config in enumerate(top_models[:5], start=0):
                        # # Reconstruct and train the stacking ensemble
                        # estimators = [(name, best_models[name]) for name in model_config['Base Models'].split(', ')]
                        # final_estimator = best_models[model_config['Meta Model']]
                        # stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=final_estimator)
                        # stacking_regressor.fit(X_train, y_train)
                        stacking_regressor = top_models[index-1]['Trained_Model']
                        y_pred = stacking_regressor.predict(X_test)
                        # Calculate and store the metric value
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

# Generate all non-empty combinations of best models
def train_stacking_models(best_models):
        model_combinations = []
        for r in range(2,3):# len(best_models) + 1):
                model_combinations.extend(list(combinations(best_models.items(), r)))
        top_models = []
        # Iterate through each combination to use each model as the meta classifier
        for combo in model_combinations:
                for meta_model_name, meta_model in combo:
                        base_learners = [(name, model) for name, model in combo if name != meta_model_name]
                        base_learner_names = ', '.join([name for name, _ in base_learners])

                        # Ensure we have at least one base learner before proceeding
                        if base_learner_names:
                                cv_results, stacking_regressor = evaluate_stacking_ensemble(base_learners, meta_model, X_train, y_train, cv= cv)
                                # Store model configuration and score
                                model_config = {
                                        'Meta Model': meta_model_name,
                                        'Base Models': base_learner_names,
                                        'Mean_Score': cv_results.mean(),
                                        'cv_results' : cv_results,
                                        'Trained_Model' : stacking_regressor
                                }
                                # Add to top models list and keep it sorted
                                top_models.append(model_config)
                                top_models = sorted(top_models, key=lambda x: x['Mean_Score'], reverse=True)[:5]  # Keep only top 5

        for i in range(len(top_models)):
                Trained_Model = top_models[i]['Trained_Model']
                joblib_file_path = os.path.join(results_folder, 'Models' , "Model" + str(i) + "_Stacked_model.joblib")
                os.makedirs(os.path.join(results_folder, 'Models'), exist_ok=True)
                joblib.dump(top_models[i]['Trained_Model'], joblib_file_path)
        
                csv_file_path = os.path.join(results_folder, 'Results' , "Model" + str(i) + "_Stacked_model.csv")
                os.makedirs(os.path.join(results_folder, 'Results'), exist_ok=True)
                results_df = pd.DataFrame(top_models[i]['cv_results'])
                results_df.to_csv(csv_file_path, index=False)

        top_models_df = pd.DataFrame(top_models)
        csv_filename = "top_5_stacking_models.csv"
        csv_file_path = os.path.join(results_folder, csv_filename)
        top_models_df.to_csv(csv_file_path, index=False)
        print("Top 5 stacking models' details have been saved to 'top_5_stacking_models.csv'")


        plot_models_comparison(top_models)

grids, grid_dicts = Pipeline_models()

scores = {metric: {model_name: [] for model_name in grid_dicts.values()} for metric in evaluation_metrics.keys()}    
best_models, scores = get_best_model(grids, grid_dicts)
# Create bar charts to visualize the evaluation metrics for each model
fig, axes = plt.subplots(len(evaluation_metrics), 1, figsize=(8, 12))
plt.subplots_adjust(hspace=0.5)
for i, (metric_name, metric_scores) in enumerate(scores.items()):     
        ax = axes[i]
        #print(metric_scores.keys())
        print(metric_name)
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

#train_stacking_models(best_models)