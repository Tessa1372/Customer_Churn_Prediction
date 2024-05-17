import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_churn_rate(reduced_data):
    
    return reduced_data['Churn'].mean()

def visualize_churn_distribution(reduced_data, column):

    sns.countplot(x=column, hue='Churn', data=reduced_data)
    plt.title(f'Churn Distribution across {column}')
    plt.show()

def analyze_correlations(reduced_data):

    corr_matrix = reduced_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()
