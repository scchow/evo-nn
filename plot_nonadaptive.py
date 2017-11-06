import matplotlib.pyplot as plt
import os
import numpy as np
import itertools
from matplotlib.ticker import FuncFormatter, MaxNLocator


def main():
    epochs = 3000
    nights = 10
    capacity = 10
    numTrials = 20
    maxAgents = 100
    # date = ("2017-11-03_08-14-15") # discount = 0.9
    date = ("2017-11-03_10-02-19") # discount = 0

    numDisabled = [0, 20, 50, 60, 70, 90]
    baseResultsPath = os.path.join("build", "Results", date, "MultiNightBarQ", "non-adaptive")

    paths = map(lambda x: os.path.join(baseResultsPath, str(maxAgents)+"_agents", str(x)+"_disabled"), numDisabled)

    dataDict = {}

    # loop through each variation
    for i in range(len(paths)):
        nDisabled = numDisabled[i]
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
        dataDict[nDisabled] = meanStd

    # Plot Performance vs Num Epochs for each variation
    numplots = len(numDisabled)
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
    for key in numDisabled:
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
    ax.get_yaxis().set_tick_params(which='both', direction='out')
    # major ticks every 20, minor ticks every 5                                      
    major_ticks = np.arange(0, 101, 20)                                              
    minor_ticks = np.arange(0, 101, 5)

    # ax.set_yticklabels(major_ticks,fontsize=20)                                               
                                       
    # ax.set_yticks(major_ticks)                                                       
    # ax.set_yticks(minor_ticks, minor=True) 
    # ax.grid(which='both')                                                            

    # or if you want differnet settings for the grids:                               
    # ax.grid(which='minor', alpha=0.2)                                                
    # ax.grid(which='major', alpha=0.5)   

    plt.yticks(range(0,100,10))
    plt.title("Performance vs Number of Epochs for " + str(nights) + " Nights of " + str(capacity) + " Capacity with " + str(maxAgents) + " Agents")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Performance (max 100)")
    plt.ylim([0,110])
    plt.yticks(np.arange(0, 110, 10))
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), title="Number of Agents\nNot Learning")
    plt.show()  



if __name__ == '__main__':
    main()