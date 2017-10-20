import matplotlib.pyplot as plt
import os
import numpy as np
import itertools

def main():
    baseResultsPath = os.path.join("build", "Results", "MultiNightBar", "10_nights", "1000_epochs", "100_agents")
    variations = [0, 10, 30, 50, 70, 90]
    # variations = [0, 90, 50]
    paths = map(lambda x: os.path.join(baseResultsPath, str(x)+"_disabled", "D"), variations)

    dataDict = {}

    # loop through each variation
    for i in range(len(paths)):
        numAgentsNotLearning = variations[i]
        path = paths[i]

        csvFname = "performance_vs_epoch.csv"
        # get CSV from the three trials
        trial0 = np.genfromtxt(os.path.join(path, "trial_0", csvFname), delimiter=",")
        trial1 = np.genfromtxt(os.path.join(path, "trial_1", csvFname), delimiter=",")
        trial2 = np.genfromtxt(os.path.join(path, "trial_2", csvFname), delimiter=",")

        trialData = np.zeros((len(trial0), 3))

        trialData[:,0] = trial0[:,1]
        trialData[:,1] = trial1[:,1]
        trialData[:,2] = trial2[:,1]

        # compute Mean and std deviation and store in numpy array
        data = np.zeros((len(trial1), 3))
        data[:,0] = trial0[:,0]
        data[:,1] = np.mean(trialData, axis=1)
        data[:,2] = np.std(trialData, axis=1)

        print data

        # associate array with number of agents not learning in dictionary
        dataDict[numAgentsNotLearning] = data

    # Plot Performance vs Num Epochs for each variation
    colors = ["r", "g", "b", "c", "m", "k"]
    markers = itertools.cycle(('o', 'v', 'x', 's', 'p', '^', '<', '>'))
    ax = plt.gca()
    i = 0
    for key in variations:
        value = dataDict[key]
        x_axis = value[:,0][::50]
        y_axis = value[:,1][::50]
        errors = value[:,2][::50]
        color = next(ax._get_lines.color_cycle)
        plt.errorbar(x_axis, y_axis, errors, linestyle='solid', marker=markers.next(), markerfacecolor=color, markeredgecolor=color, c=color, label=str(key), mew=5.0)
        i+=1

    handles, labels = ax.get_legend_handles_labels()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])


    plt.title("Performance vs Number of Epochs")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Performance (max 100)")
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), title="Number of Agents Not Learning")
    plt.show()  



if __name__ == '__main__':
    main()