import seaborn as sns
import matplotlib.pyplot as plt


def Numerical_plots(df, col, type):
    if type.lower()=='histogram':
        sns.set_style('darkgrid')
        plt.title('Histogram for: ' + str(col))
        hist = sns.histplot(data=df, x=col)
        plt.show()
    
    if type.lower()=='box':
        sns.set_style('darkgrid')
        plt.title('Box Plot for: ' + str(col))
        box = sns.boxplot(data=df, x=col)
        plt.show()

def Categorical_plots(df, col):
    print(f"Category: {col}")
    sns.set_style('darkgrid')
    plt.title('CountPlot for: ' + str(col))
    count_plt = sns.countplot(data=df, x=col)
    plt.show()
    