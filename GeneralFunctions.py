import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def readFile(file):
    try:
        df = pd.read_csv(file)
        return df
    except FileNotFoundError:
        print("File not found")
        sys.exit()

def showCorrelation(df,title, method):
    corr = df.corr(method = method)
    graph = sns.heatmap(corr, annot=True, cmap="Blues")
    plt.title(title)
    plt.setp(graph.get_xticklabels(), rotation=15)
    plt.show()