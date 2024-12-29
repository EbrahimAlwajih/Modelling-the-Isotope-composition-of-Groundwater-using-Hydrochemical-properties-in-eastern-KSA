import numpy as np
import matplotlib.pyplot as plt

# # Sample Data
# observed =	[-20.95,	-21.19,	-20.87,	-21.45,	-20.54,	-17.21,	-23.84,	-20.24,	-27.97,	-22.54]


# models = { 
# 'SVR':	    [-20.65109211,	-21.20549315,	-20.7510044,	-20.83592049,	-20.98843335,	-21.4409898,	-22.01751712,	-23.86781586,	-27.36404707,	-23.09769981],
# 'KNN':	    [-21.794,   	-21.852,    	-20.994,    	-22.97,      	-20.76,     	-20.454,	-22.85,	-21.088,	-28.436,	-23.186],
# 'RF':	    [-21.9035,  	-23.7057,   	-20.8211,   	-22.0357,   	-20.849,	-20.1511,	-21.7886,	-21.9073,	-28.4361,	-23.1106],
# 'Ext':	    [-21.414,   	-23.2133,   	-20.8411,   	-22.1395,   	-20.9297,	-19.4117,	-23.2275,	-20.9322,	-28.6252,	-23.9064],
# 'AdaBt':	[-21.156,   	-26.41571429,   -21.14222222,	-21.156,    	-21.14222222,	-20.504,	-21.2425,	-22.14875,	-28.5,	-22.66],
# 'Grd':	    [-20.89690913,	-26.12561313,	-20.75879693,	-20.69947369,	-20.75879693,	-19.55562787,	-21.60343584,	-20.88805057,	-28.394469,	-23.88155181],
# 'Bag':	    [-22.534,   	-23.91,     	-21.47,     	-21.852,    	-20.834,	-20.319,	-21.981,	-21.997,	-28.174,	-23.938],
# 'CAT':	    [-21.14375406,	-25.45952815,	-20.61943462,	-21.36035389,	-20.88473182,	-19.97346556,	-22.5190788,	-20.94629424,	-28.55959029,	-23.70974279],
# 'Model0':	[-20.91843197,	-21.40444063,	-21.09891225,	-21.74882407,	-21.22673046,	-18.28103194,	-23.04884245,	-20.88860135,	-28.10198211,	-21.53063821],
# 'Model1':	[-20.7685141,	-21.6756648,	-19.90207233,	-22.51584542,	-19.95279637,	-18.54627555,	-22.49751393,	-20.63164276,	-27.42358004,	-23.23572356],
# 'Model2':	[-21.81083333,	-21.41916667,	-20.81333333,	-21.74571429,	-20.89666667,	-18.07,	-21.81083333,	-21.68166667,	-28.37,	-21.41916667],
# 'Model3':	[-21.25842105,	-20.95571429,	-20.62,       	-21.25842105,	-20.62642857,	-19.62,	-22.31857143,	-20.91909091,	-28.612,	-23.38615385],
# 'Model4':	[-20.5882,  	-20.9461,   	-20.2386,   	-22.3283,   	-20.2412,	-19.0561,	-21.704,	-20.3575,	-28.1704,	-23.4914],
# }
import pandas as pd
import os
dataset_name = "δ18O"
results_folder = os.path.join('J:\\My Drive\\Research\\AI Reaserch Team - KFUPM\\code_v3\\Best Models\\V2\\results_' + dataset_name +'_(3 att_v2)','y_pred.csv')
# directory_path = os.path.join(results_folder, 'y_pred.csv')  
# Read the CSV file into a DataFrame
df = pd.read_csv(results_folder)
observed = df.iloc[:, 1]
df = df.iloc[:, 2:]
# Initialize an empty dictionary to store the columns
models = {}

# Iterate over the DataFrame columns
for column in df.columns:
    # Save each column in the dictionary with the column name as the key
    models[column] = df[column].values


markers = ['s', 'k+', '^', 'D', 'x', '*', "p", "P", "X", "d", "h","1","o"]
marker_size = 10
std_dev_obs = np.std(observed)
correlations = {model: np.corrcoef(observed, prediction)[0,1] for model, prediction in models.items()}
std_devs = {model: np.std(prediction) for model, prediction in models.items()}

# Plotting
fig = plt.figure(dpi=100)
plt.rcParams.update({'font.size': 14})
ax1 = fig.add_subplot(111, polar=True)

# Different markers for each model
# Plot each model with different marker
for i, (model, prediction) in enumerate(models.items()):
    corr_radians = np.arccos(correlations[model])
    ax1.plot(corr_radians, std_devs[model], markers[i], label=model, markersize=marker_size)  

# Plot observed data with a different marker  43.53405620924243
ax1.plot(0, std_dev_obs, 'o', label='Observed', markersize=marker_size)   # Plus marker for observed

# Set labels and titles for the polar plot
correlation_values = np.arange(0, 1.01, 0.1)  # From 0 to 1 with an interval of 0.1
correlation_values_additional = np.arange(0.9, 1.01, 0.02)  # Additional correlation values from 0.9 to 1 with an interval of 0.01
correlation_values_all = np.concatenate([correlation_values, correlation_values_additional])  # Combine both sets of correlation values
correlation_radians = np.arccos(correlation_values_all)

# Set ticks and labels
ax1.set_xticks(correlation_radians)
ax1.set_xticklabels([f'{corr:.2f}' for corr in correlation_values_all], rotation=45,fontsize=15)  # Format to display two decimal places and rotate labels for better readability

ax1.set_title('Taylor Diagram', pad=20,fontsize=20)

# Limit the plot to the first quadrant
ax1.set_thetamin(0)
ax1.set_thetamax(90)

# Axis labels
ax1.set_xlabel('Standard Deviation of Predicted Values', labelpad=20,fontsize=20)
ax1.set_ylabel('Standard Deviation of Observed Values', labelpad=40,fontsize=20)

# Adjust the grid for RMS contours to better fit the plot
max_std_dev = max([std_dev_obs] + list(std_devs.values()))
corr_grid, std_grid = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, max_std_dev * 1.5, 100))
rms_grid = np.sqrt(std_grid**2 + std_dev_obs**2 - 2 * std_grid * std_dev_obs * corr_grid)
# Convert correlation grid to radians for polar plot
corr_grid_radians = np.arccos(corr_grid)

# Plot RMS contours in cyan color
# #fefefe #0c7268 #cfe3e0 #629883 #9dc3bb
contours = ax1.contour(corr_grid_radians, std_grid, rms_grid, levels=5, colors='Blue')
ax1.clabel(contours, inline=True, fontsize=20)

# Add a circle of radius std_dev_obs
circle_radius = std_dev_obs 
theta = np.linspace(0, np.pi/2, 100)
r = np.full_like(theta, circle_radius)
ax1.plot(theta, r, 'k--')  # Red dashed line for the circle contour

# Add label for Standard Deviation 
correlation_label = f'Standard Deviation: {round(circle_radius, 2)}'
ax1.text(np.pi/4, circle_radius, correlation_label, rotation=-45, horizontalalignment='center', verticalalignment='center', fontsize=15)
ax1.text(np.pi/4, std_grid.max() + 0.2, 'Correlation Coefficient', rotation=-45, horizontalalignment='center', verticalalignment='center', fontsize=20)

# Add a legend
plt.legend(loc='upper right', bbox_to_anchor=(1, 1),fontsize=10)

# # Find the maximum standard deviation
# max_std_dev = max(std_devs.values())
# # Convert dictionary values to a list
# std_values = list(std_devs.values())

# # Calculate the standard deviation of the std_values list
# std_dev_of_std_devs = np.std(std_values)

# # Set axis limits
# ax1.set_ylim(0, max_std_dev + 2 * np.abs(std_dev_of_std_devs))
# ax1.set_xlim(0, np.pi/2)  # 0 to pi/2 for x-axis covers the entire quadrant

# Show plot
plt.show()
plt
