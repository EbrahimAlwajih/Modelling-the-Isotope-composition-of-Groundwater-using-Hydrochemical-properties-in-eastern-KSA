import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dataset_name = "δ2H&δ18O"  
url = '.\\'+dataset_name+'.csv'
df = pd.read_csv(url, header=0)
#print(df.head())  
target_column = dataset_name + ' (‰)'
n_rows = 5
n_cols = 4

# Create a figure and a grid of subplots
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 10))
axes = axes.flatten()

# Plot each column in the DataFrame in its own subplot
for i, column in enumerate(df.columns):
    df[column].hist(bins=10, ax=axes[i], color='#50705c')
    axes[i].set_title(column)

# Hide any unused subplots
for i in range(len(df.columns) + 1, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig('.\\Figures'+dataset_name+'\\Hist.svg', dpi=150)  
# plt.show()
plt.close()

describe = df.describe(include='all')
describe.to_excel('describe.xlsx')
#print(describe)

correlation_matrix = df.corr()[target_column]
correlation_matrix = correlation_matrix.sort_values(ascending=False)
correlation_matrix = correlation_matrix.drop(target_column)
# Plot the correlation between all attributes and the target using a bar plot
plt.figure(figsize=(8, 5),dpi=150)  # Adjust the figure size as needed
plt.rcParams.update({'font.size': 12})
sns.barplot(x=correlation_matrix.index[:], y=correlation_matrix.values[:], palette='Spectral')
plt.xticks(rotation=90)
plt.tight_layout()
plt.xlabel('Attributes')
plt.ylabel('Correlation')
plt.title('Correlation between Attributes and Target')
plt.savefig('.\\Figures'+dataset_name+'\\corr.svg', dpi=150)  
plt.show()
plt.close()

# Assuming 'df' is your DataFrame and the correlation matrix calculation has been done
correlation_matrix = df.corr()
correlation_values = correlation_matrix.values
attribute_names = df.columns  # Get the attribute names from the DataFrame
n = correlation_values.shape[0]

# Replace upper triangle with NaN
for i in range(n):
    for j in range(i+1, n):
        correlation_values[i, j] = np.nan
# Plotting setup
#plt.figure(figsize=(30, 20))
plt.figure(figsize=(8, 5),dpi=150)
plt.rcParams.update({'font.size': 10})
heatmap = plt.imshow(correlation_values, cmap='Spectral', aspect='auto')
plt.colorbar(heatmap)

# Add text annotations for correlation values
for i in range(correlation_values.shape[0]):
    for j in range(correlation_values.shape[1]):
        if not np.isnan(correlation_values[i, j]):  # Only add text if not NaN
            plt.text(j, i, f'{correlation_values[i, j]:.2f}', ha="center", va="center", color="w")

# Setting attribute names as tick labels
plt.xticks(ticks=np.arange(len(attribute_names)), labels=attribute_names, rotation=90)
plt.yticks(ticks=np.arange(len(attribute_names)), labels=attribute_names)
plt.tight_layout()
plt.title('Single Corner Correlation Heatmap')
plt.show()
print('')