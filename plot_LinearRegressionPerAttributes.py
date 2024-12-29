import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from pandas import read_csv

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep].astype(np.float64)

# Generating a mock dataset
dataset_name = "δ2H"  
url = '.\\'+dataset_name+'.csv'
dataset = read_csv(url, header=0)
dataset = clean_dataset(dataset)
input_vars = dataset.columns[:-1]


def scatter_hist(x, y, ax, ax_histx, ax_histy, var_name, output_name):
    # the scatter plot:
    color = '#996633'
    ax.scatter(x, y, color=color, label='Data Points', marker ='+')
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        #label.set_fontsize('large')  # Set the font size
        label.set_fontweight('bold')  # Set the font weight to bold
    # Fit linear regression
    model = LinearRegression(fit_intercept=True)
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    ax.plot(x, y_pred, color='red', label='Regression Line')
    ax.set_title(f'Regression Line for {var_name} vs Y')
    ax.legend(loc='lower left')
    # Setting axis labels
    ax.set_xlabel(var_name, fontdict=dict(weight='bold'))
    ax.set_ylabel(output_name, fontdict=dict(weight='bold'))

    # histograms
    binwidth = 5
    ax_histx.hist(x, bins=40, edgecolor = "black", linewidth=0.75, color=color)
    ax_histy.hist(y, bins=40, orientation='horizontal', edgecolor = "black", linewidth=0.75, color=color)

    # Remove labels from the histograms (redundant axis values)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
import os
# Plot for each input variable
for var in input_vars:
    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.00, hspace=0.00)

    # Create the Axes
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    # Remove axis lines and ticks for the histograms
    ax_histy.spines[['right', 'top', 'bottom']].set_visible(False)
    ax_histx.spines[['right', 'top', 'left']].set_visible(False)
    ax_histy.set_xticks([])
    ax_histx.set_yticks([])
    # Draw the scatter plot and marginals
    scatter_hist(dataset[var].values, dataset[dataset_name + ' (‰)'].values, ax, ax_histx, ax_histy, var, dataset_name + ' (‰)')
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    var1 = var.replace("/"," ")
    svg_path = os.path.join('.\\Figures'+dataset_name,  var1 + '.svg')
    plt.savefig(svg_path)
    #plt.show()