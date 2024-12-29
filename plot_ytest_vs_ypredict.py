import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Define the dataset name and path to the results file
dataset_name = "δ2H"
results_folder = os.path.join('J:\\My Drive\\Research\\AI Reaserch Team - KFUPM\\code_v3\\Best Models\\V2\\results_' + dataset_name + '_(3 att_v2)', 'y_pred.csv')
df = pd.read_csv(results_folder)
observed = df.iloc[:, 1]
df = df.iloc[:, 2:]
models = {}

for column in df.columns:
    models[column] = df[column].values

plt.figure(figsize=(10, 8))
# Set font properties for bold labels
rcParams['axes.labelweight'] = 'bold'
rcParams['xtick.labelsize'] = 30  # Set size for x-tick labels if needed
rcParams['ytick.labelsize'] = 30  # Set size for y-tick labels

sns.set(style="whitegrid")

for model_name, predictions in models.items():
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")
    sns.scatterplot(x=observed, y=predictions,s=100)  # Scatter plot for each model
    plt.plot([observed.min(), observed.max()], [observed.min(), observed.max()], color='red', lw=2)  # Line of perfect prediction
    # plt.tick_params(axis='both', labelsize=20, width=3, labelcolor='black')  # Bold and larger y-axis tick labels
    plt.title(f'Regression Model Performance for {model_name}',fontsize=30,fontweight='bold' )
    plt.xlabel('Actual Values', fontsize=30, fontweight='bold')
    plt.ylabel('Predicted Values', fontsize=30, fontweight='bold')
    plt.legend([ 'Predictions', 'Ideal'], prop={'weight':'bold', 'size' : 20})

    # Apply bold and larger font style to tick labels
    ax = plt.gca()  # Get current axis
    ax.tick_params(axis='both', which='major', labelsize=20, width=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    svg_path = os.path.join('.\\Figures'+ dataset_name, 'ytest_vs_ypredict', model_name + '.svg')
    plt.savefig(svg_path)#, dpi=100, figsize=(4.1,3))



plt.show
