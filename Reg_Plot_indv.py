import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Clear the current figure and load the data
plt.clf()
y_test = np.load('y_test.npy')  # Assuming y_test.mat is converted to y_test.npy
filename = 'Predection_Results.xlsx'
data = pd.read_excel(filename)

# Data indices
indices = [0, 1, 2, 3, 4, 5, 8, 10, 12, 14, 16, 18]  # Python uses 0-based indexing
titles = ['Support Vector Regression', 'Decision Tree Regression', 'Linear Regression',
          'Neural Network Regression', 'Gradient Boosting Regression', 'Random Forest Regression',
          'Stacked Regression Model1', 'Stacked Regression Model2', 'Stacked Regression Model3',
          'Stacked Regression Model4', 'Stacked Regression Model5', 'Stacked Regression Model6']
import os
def scatter_plot(actual, predicted, title_text):
    plt.scatter(actual, predicted, c='b', marker='o', label='Predicted vs Actual')
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r-', linewidth=2)
    plt.xlabel('Actual',fontdict=dict(weight='bold',fontsize=15))
    plt.ylabel('Predicted',fontdict=dict(weight='bold',fontsize=15))
    plt.title(title_text, fontdict=dict(weight='bold',fontsize=15))
    plt.gca().tick_params(axis='both', labelsize=15)
    svg_path = os.path.join('..\\Figures', 'Reg_Plot_' + title_text + '.svg')
    plt.savefig(svg_path)
    plt.show()

for i in indices:
    scatter_plot(y_test, data.iloc[:, i], titles[indices.index(i)])
