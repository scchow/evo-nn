#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Eigen>
#include <ctime>

#include "Domains/MultiNightBarQ.h"

using std::vector;
using std::string;
using namespace Eigen;

int runMultiTrials(char* timeStr, size_t numAgents, size_t nAgentsDisabled, int trialNum, std::vector<int> barPadding, int adaptiveLearning, size_t maxEpoch, double temp){
    std::cout << "Testing MultiNightBarQ class in MultiNightBarQ.h\n";
    
    size_t capacity = 10;
    size_t numNights = 10;

    string evalFunc = "D";
    // if (adaptiveLearning == 8){    
    if (adaptiveLearning == 0 or adaptiveLearning == 8){
        evalFunc = "G";
    }

    double learningRate = 0.1;
    double discount = 0.0;
    double epsilon = 0.1;
    double maxReward = 10; // set max reward to 10 since agents get D as reward

    size_t nEps = maxEpoch;

    // std::vector<int> barPadding = {};

    std::cout << "This program will use Q-learning to train " << numAgents << "-agent team over " << nEps << " learning epochs\n";
    std::cout << nAgentsDisabled << " Agents will be not be learning\n";
    std::cout << "Adaptive Training is set to " << adaptiveLearning <<"\n";
    std::cout << " Temperature (only matters for softmax): " << temp << "\n";
    std::cout << "Agent Q-Learning parameters:\n";
    std::cout << "  Learning Rate: " << learningRate << "\n";
    std::cout << "  Discount: " << discount << "\n";
    std::cout << "  Epsilon: " << epsilon << "\n";
    std::cout << "  Evaluation function: " << evalFunc << "\n";
    std::cout << "Environment parameters:\n";
    std::cout << "  Bar capacity: " << capacity << "\n";
    std::cout << "  Number of nights: " << numNights << "\n";
    std::cout << "Using learning Rate (alpha) only Q-Learning";

    
    // int trialNum;
    // std::cout << "Please enter trial number [NOTE: no checks enabled to prevent overwriting existing files, user must make sure trial number is unique]: ";
    // std::cin >> trialNum;
    
    MultiNightBarQ trainDomain(numNights, capacity, barPadding, numAgents, evalFunc, 
                                 learningRate, discount, epsilon, maxReward, nAgentsDisabled, adaptiveLearning, maxEpoch, temp);

    int buffSize = 500;
    char fileDir[buffSize];

    sprintf(fileDir,"Results/%s/MultiNightBarQ/%s%s/%d_agents/%d_disabled/trial_%d",
            timeStr, 
            adaptiveLearning == 0 ? "non-adaptive" : 
                (adaptiveLearning==1 ? "adaptive_max_n" :
                (adaptiveLearning==2 ? "adaptive_max_centralized" : 
                (adaptiveLearning==3 ? "adaptive_softmax_G-distributed/temp_" : 
                (adaptiveLearning==4 ? "adaptive_softmax_G-localized/temp_" :
                (adaptiveLearning==5 ? "adaptive_softmax_G-centralized/temp" :
                (adaptiveLearning==6 ? "adaptive_softmax_D/temp_" :
                (adaptiveLearning==7 ? "fixed_prob_learning/prob_" :
                (adaptiveLearning==8 ? "adaptive_softmax_G-only/temp_" :
                                    "unknown")))))))),
            adaptiveLearning >=3 ? std::to_string(temp).c_str() : "",
            (int)numAgents,
            (int)nAgentsDisabled,
            trialNum
            );

    char mkdir[buffSize];
    sprintf(mkdir,"mkdir -p %s",fileDir);
    system(mkdir);
    
    std::cout << "\nWriting log files to: " << fileDir << "\n\n";
    
  // Initialize Logging Results

    // Get path of results file
    char resultsFilePath[buffSize];
    sprintf(resultsFilePath,"%s/results.csv",fileDir);

    std::stringstream resultsStream;

    resultsStream << resultsFilePath;

    // open ofstream to write to file
    std::ofstream resultsFile;
    resultsFile.open(resultsStream.str().c_str(),std::ios::app);

    char actionsFilePath[buffSize];
    sprintf(actionsFilePath,"%s/actions.csv",fileDir);

    char qTableFilePath[buffSize];
    sprintf(qTableFilePath,"%s/qTables.csv",fileDir);

    char agentActionsFilePath[buffSize];
    sprintf(agentActionsFilePath,"%s/agent_actions.csv",fileDir);

    // char cFile[buffSize];
    // sprintf(cFile,"%s/capacities.txt",fileDir);
    char readmeFile[buffSize];
    sprintf(readmeFile, "%s/readme.txt",fileDir);
    // char pveFile[buffSize];
    // sprintf(pveFile, "%s/performance_vs_epoch.csv",fileDir);

    char numLearningFile[buffSize];
    sprintf(numLearningFile, "%s/numLearning.csv",fileDir);

    // trainDomain.OutputPerformance(eFile);
    trainDomain.outputParameters(readmeFile, nEps);

    std::cout << "Beginning Training\n";
    double G;
    trainDomain.initialiseEpoch();
    for (size_t n = 0; n < nEps; n++){
        std::cout << "Episode " << n << "...\n";
        trainDomain.simulateEpoch(n);
        trainDomain.outputNumLearning(numLearningFile, n);
        G = trainDomain.computeFinalScore();
        std::cout << "Score " << G << std::endl; 

        // Log results of epoch
        resultsFile << n << ", " << G << "\n";
   
    }
    std::cout << "Final Score" << std::endl;
    G = trainDomain.computeFinalScoreOutput(qTableFilePath, actionsFilePath);
    std::cout << "Score " << G << std::endl; 
    trainDomain.outputAgentActions(agentActionsFilePath);
    // char NNFile[buffSize];
    // sprintf(NNFile,"%s/NNs.txt",fileDir);
    
    // std::cout << "\nWriting final control policies to file...\n";
    
    // trainDomain.OutputControlPolicies(NNFile);
    
    // int isTest;
    // std::cout << "Please enter [0] to end program, [1] to test stored NN policies in new environment: ";
    // std::cin >> isTest;
    // if (isTest < 0 || isTest > 1){
    //   std::cout << "Input is out of range. Exiting program.\n";
    //   isTest = 0;
    // } 
    
    // if (isTest == 1){
      
    //   capacity = 10;
    //   numNights = 10;
    //   std::cout << "Test parameters:\n";
    //   std::cout << "  Bar capacity: " << capacity << "\n";
    //   std::cout << "  Number of nights: " << numNights << "\n";
      
    //   std::cout << "Testing stored control policies on new world...\n";
      
    //   MultiNightBar testDomain(numNights, capacity, nPop, evalFunc, agents, nAgentsDisabled);

    //   char eeFile[buffSize];
    //   sprintf(eeFile,"%s/results_test.txt",fileDir);
    //   char aaFile[buffSize];
    //   sprintf(aaFile,"%s/actions_test.txt",fileDir);
    //   char ccFile[buffSize];
    //   sprintf(ccFile,"%s/bars_test.txt",fileDir);
      
    //   testDomain.ExecutePolicies(NNFile, aaFile, ccFile, eeFile, nInputs, nOutputs, nHidden);
    // }
    
    std::cout << "Trial" << trialNum << " complete!\n";
    resultsFile.close();
    return 0;
}


int main(){
    size_t numTrials = 2;
    size_t maxEpoch = 50001;
    int adaptiveLearning;
    std::vector<size_t> numAgentVariations = {100};

    std::vector<int> barPadding = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    // In fixed probability learning, temperature repurposed to be probability of an agent learning
    // std::vector<double> temps = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    std::vector<double> temps = {100,10,50,500};
    
    // Adaptive Learning types
    // 0 - No Learning
    // 1 - Select Max n agents
    // 2 - Centralized Max - Probability = Largest Normalize by largest Impact
    // 3 - SoftMax with dG/dpi as impact with temperature - Distributed (no normalization)
    // 4 - SoftMax with dG/dpi as impact with temperature - Local (normalization by impacts of current night)
    // 5 - SoftMax with dG/dpi as impact with temperature - Centralized (normalization by largest impact)
    // 6 - SoftMax with dD/dpi as impact

    /// flag to use adaptive learning
    /// 0 - no adaptive learning
    /// 1 - adaptive learning using fixed max
    /// 2 - adaptive learning using softmax of G only (no temperature) - Distributed verion (no normalization)
    /// 3 - adaptive learning using softmax of G with temperature - Distributed verion (no normalization)
    /// 4 - adaptive learning using softmax of G with temperature - Local communication verion (normalization with same night)
    /// 5 - adaptive learning using softmax of G with temperature - Centralized verion (normalization across all)
    /// 6 - adaptive learning using softmax of D with temperature
    /// 7 - fixed probability learning
    /// 8 - adaptive learning using softmax of G with temperature - Distributed verion, learning with G and impact with G (onlyG)

    // std::vector<int> adaptiveLearningSchemes = {0,3,4,5};
    std::vector<int> adaptiveLearningSchemes = {8};

    // Get timestamp
    time_t rawtime;
    struct tm * timeinfo;
    char timeStr[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(timeStr,sizeof(timeStr),"%Y-%m-%d_%H-%M-%S", timeinfo);

    char* folderName = (char *)"onlyG2";

    for (size_t schemeInd = 0; schemeInd < adaptiveLearningSchemes.size(); ++schemeInd){
        adaptiveLearning = adaptiveLearningSchemes[schemeInd];
        for (size_t k = 0; k < numAgentVariations.size(); ++k){
            size_t numAgents = numAgentVariations[k];
            // If using softmax, specifying number of agents to not learn doesn't matter

            if (adaptiveLearning == 0 or adaptiveLearning==1)
                for (size_t i = 0; i < 20; ++i){
                size_t numDisabled = i*20; 
                for (size_t j = 0; j < numTrials; ++j){
                    if (numAgents > numDisabled){
                        // runMultiTrials(timeStr, numAgents, numDisabled, j, barPadding, adaptiveLearning, maxEpoch, 0);
                        runMultiTrials(folderName, numAgents, numDisabled, j, barPadding, adaptiveLearning, maxEpoch, 0);
                    }
                }
            }
            else{
                for (size_t t = 0; t < temps.size(); ++t){
                    double temp = temps[t];
                    for (size_t j = 0; j < numTrials; ++j){
                        // runMultiTrials(timeStr, numAgents, 0, j, barPadding, adaptiveLearning, maxEpoch, temp);
                        runMultiTrials(folderName, numAgents, 0, j, barPadding, adaptiveLearning, maxEpoch, temp);
                    }
                }
            }
        }
    }
    
}


////Testing with padding
    // for (size_t j = 0; j < numTrials; ++j){
    //     size_t numAgents = 90;
    //     size_t numDisabled = 0;
    //     int trialNum = j;
    //     std::vector<int> barPadding = {1, 1, 0, 0, 2, 1, 2, 1, 0, 2};
    //     runMultiTrials(numAgents, numDisabled, trialNum, barPadding);
    // }


