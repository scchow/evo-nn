import matplotlib.pyplot as plt
import os
import numpy as np
import itertools

def main():
    epochs = 10000
    nights = 10
    capacity = 10
    variations = [0, 10, 30, 50, 90]
    numTrials = 20
    numAgents = 100
    baseResultsPath = os.path.join("build", "Results", "MultiNightBarQ", str(nights)+"_nights", str(epochs)+"_epochs", str(numAgents) + "_agents")

    # variations = [0, 90, 50]
    paths = map(lambda x: os.path.join(baseResultsPath, str(x)+"_disabled", "D"), variations)

    dataDict = {}

    # loop through each variation
    for i in range(len(paths)):
        numAgentsNotLearning = variations[i]
        path = paths[i]

        csvFname = "results.csv"
        
        trialFolders = ["trial_"+str(i) for i in range(numTrials)]
        # get first CSV to format data array
        trial0 = np.genfromtxt(os.path.join(path, "trial_0", csvFname), delimiter=",")
        data = np.zeros((len(trial0), numTrials))

        for i in range(numTrials):
            trialData = np.genfromtxt(os.path.join(path, trialFolders[i], csvFname), delimiter=",")
            data[:,i] = trialData[:,1]

        # compute Mean and std deviation and store in numpy array
        meanStd = np.zeros((len(trial0), 3))
        meanStd[:,0] = trial0[:,0]
        meanStd[:,1] = np.mean(data, axis=1)
        meanStd[:,2] = np.std(data, axis=1)

        # print data

        # associate array with number of agents not learning in dictionary
        dataDict[numAgentsNotLearning] = meanStd

    # Plot Performance vs Num Epochs for each variation
    numplots = len(variations)
    # colormap = plt.cm.gist_ncar
    plt.style.use('ggplot')
    # plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, numplots))))

    # colors = ["r", "g", "b", "c", "m", "k"]
    markers = itertools.cycle(('o', 'v', 'x', 's', 'p', '^', '<', '>'))
    ax = plt.gca()
    i = 0
    increment = 500
    for key in variations:
        value = dataDict[key]
        x_axis = value[:,0][::increment]
        y_axis = value[:,1][::increment]
        errors = value[:,2][::increment]
        # color = next(ax._get_lines.color_cycle)
        # plt.errorbar(x_axis, y_axis, errors, linestyle='solid', marker=markers.next(), markerfacecolor=color, markeredgecolor=color, c=color, label=str(key), mew=5.0)
        plt.errorbar(x_axis, y_axis, errors, linestyle='solid', marker=markers.next(), label=str(key), mew=5.0)
        i+=1

    handles, labels = ax.get_legend_handles_labels()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    plt.yticks(range(0,100,10))
    plt.title("Performance vs Number of Epochs for " + str(nights) + " Nights of " + str(capacity) + " Capacity with " + str(numAgents) + " Agents")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Performance (max 100)")
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), title="Number of Agents Not Learning")
    plt.show()  



if __name__ == '__main__':
    main()