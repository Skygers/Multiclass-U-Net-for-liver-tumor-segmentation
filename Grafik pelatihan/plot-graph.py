import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
#plt.rcParams.update({'figure.dpi': '600'})


filename = "history.csv"
dataframe = pd.read_csv(filename)

def plot_history(dataframe, metrics=["accuracy", "val_accuracy"]):
    with plt.style.context(['science', 'ieee']):
        plt.figure(figsize=(12,6), dpi=600)
        for metric in metrics:
            plt.plot(dataframe[metrics], linewidth=3)(12,6
        plt.suptitle("Accuracy V Epoch", fontsize=20)
        plt.ylabel("Accuracy", fontsize=20)
        plt.xlabel("epoch", fontsize=20)
        plt.legend(metrics, loc='center right', fontsize=15)
        plt.show()


plot_history(dataframe)