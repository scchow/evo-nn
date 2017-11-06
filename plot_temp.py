import matplotlib.pyplot as plt
import os
import numpy as np
import itertools

def main():
    epochs = 3000
    nights = 10
    capacity = 10
    numTrials = 20
    numAgents = 100
    # date, adapt_type = ("2017-11-03_08-14-15", "softmax") # varying temperature - discount = 0.9
    # date, adapt_type = ("2017-11-03_10-02-19", "softmax") # varying temperature - discount = 0
    # date, adapt_type = ("2017-11-04_05-24-47", "softmax_D") # dD metric - discount = 0
    # date, adapt_type = ("2017-11-04_06-29-28", "softmax_D") # dD/dPi impact - discount = 0


    # Use this reordering so that the first two get same plot coloring/symbol as comparison plot
    # If plot something else, remember to change the ordering using list at the bottom
    temperatures = ["All Agents Learning", 300, 100, 500, 1000, 5000]
    # temperatures = [10, 50]
    # baseResultsPath = os.path.join("build", "Results", date, "MultiNightBarQ","adaptive_"+adapt_type)

    # paths = map(lambda x: os.path.join(baseResultsPath, "temp_"+str(x),str(numAgents)+"_agents", "0_disabled"), temperatures)

    paths = map(lambda x: os.path.join("build", "Results", "adaptive_softmax", "temp_"+str(x),str(numAgents)+"_agents", "0_disabled"), temperatures[1:])

    paths = [os.path.join("build", "Results", "2017-11-03_10-02-19", "MultiNightBarQ","non-adaptive", "100_agents", "0_disabled")] + paths

    dataDict = {}

    # loop through each variation
    for i in range(len(paths)):
        temp = temperatures[i]
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
        dataDict[temp] = meanStd

    # Plot Performance vs Num Epochs for each variation
    numplots = len(temperatures)
    # colormap = plt.cm.gist_ncar
    plt.style.use('ggplot')
    # plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, numplots))))

    # colors = ["r", "g", "b", "colors", "m", "k"]
    markers = itertools.cycle(('o', 'v', 'x', 's', 'p', '^', '<', '>'))
    linestyles = itertools.cycle(('-', '--', '-.', ':'))
    ax = plt.gca()
    i = 0
    increment = 200
    maxEpoch = 3000
    for key in temperatures:
        value = dataDict[key]
        x_axis = value[:,0][:maxEpoch:increment]
        y_axis = value[:,1][:maxEpoch:increment]
        errors = value[:,2][:maxEpoch:increment]
        # color = next(ax._get_lines.color_cycle)
        # plt.errorbar(x_axis, y_axis, errors, linestyle='solid', marker=markers.next(), markerfacecolor=color, markeredgecolor=color, c=color, label=str(key), mew=5.0)
        plt.errorbar(x_axis, y_axis, errors, linestyle=linestyles.next(), marker=markers.next(), label=str(key), mew=5.0)
        i+=1

    handles, labels = ax.get_legend_handles_labels()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])



    plt.yticks(range(0,100,10))
    plt.title("Performance vs Number of Epochs for " + str(nights) + " Nights of " + str(capacity) + " Capacity with " + str(numAgents) + " Adaptive " + "Agents")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Performance (max 100)")
    plt.ylim([0,110])
    plt.yticks(np.arange(0, 110, 10))
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    order = [0,2,1,3,4,5]
    # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='center left', bbox_to_anchor=(1, 0.5), title="Temperature")
    plt.show()  



if __name__ == '__main__':
    main()