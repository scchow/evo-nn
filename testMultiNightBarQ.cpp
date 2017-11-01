#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Eigen>
#include <ctime>

#include "Domains/MultiNightBarQ.h"

using std::vector;
using std::string;
using namespace Eigen;

int runMultiTrials(char* timeStr, size_t numAgents, size_t numDisabled, int trialNum, std::vector<int> barPadding, int adaptiveLearning, size_t maxEpoch){
    std::cout << "Testing MultiNightBarQ class in MultiNightBarQ.h\n";
    
    size_t capacity = 10;
    size_t numNights = 10;
    // size_t numAgents = 100;
    size_t nAgentsDisabled = numDisabled;
    string evalFunc = "D";

    double learningRate = 0.1;
    double discount = 0.9;
    double epsilon = 0.01;
    double maxReward = 10; // set max reward to 10 since agents get D as reward

    size_t nEps = maxEpoch;

    // std::vector<int> barPadding = {};

    std::cout << "This program will use Q-learning to train " << numAgents << "-agent team over " << nEps << " learning epochs\n";
    std::cout << nAgentsDisabled << " Agents will be not be learning\n";
    std::cout << "Adaptive Training is set to " << adaptiveLearning <<"\n";
    std::cout << "Agent Q-Learning parameters:\n";
    std::cout << "  Learning Rate: " << learningRate << "\n";
    std::cout << "  Discount: " << discount << "\n";
    std::cout << "  Epsilon: " << epsilon << "\n";
    std::cout << "  Evaluation function: " << evalFunc << "\n";
    std::cout << "Environment parameters:\n";
    std::cout << "  Bar capacity: " << capacity << "\n";
    std::cout << "  Number of nights: " << numNights << "\n";

    
    // int trialNum;
    // std::cout << "Please enter trial number [NOTE: no checks enabled to prevent overwriting existing files, user must make sure trial number is unique]: ";
    // std::cin >> trialNum;
    
    MultiNightBarQ trainDomain(numNights, capacity, barPadding, numAgents, evalFunc, 
                                 learningRate, discount, epsilon, maxReward, nAgentsDisabled, adaptiveLearning, maxEpoch);

    int buffSize = 100;
    char fileDir[buffSize];
    sprintf(fileDir,"Results/%s/MultiNightBarQ/%s/%d_nights/%d_epochs/%d_agents/%d_disabled/%s/trial_%d",
                timeStr, 
                 (adaptiveLearning==1 ? "adaptive_max" : (adaptiveLearning==2 ? "adaptive_softmax" :"static")), 
                (int)numNights,
                (int)nEps, 
                (int)numAgents,
                (int)nAgentsDisabled,
                evalFunc.c_str(),
                trialNum);

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


    // trainDomain.OutputPerformance(eFile);
    trainDomain.outputParameters(readmeFile, nEps);

    std::cout << "Beginning Training";
    double G;
    trainDomain.initialiseEpoch();
    for (size_t n = 0; n < nEps; n++){
        std::cout << "Episode " << n << "...\n";
        trainDomain.simulateEpoch(n);
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
    size_t numTrials = 20;
    size_t maxEpoch = 3000;
    int adaptiveLearning = 2;
    std::vector<size_t> numAgentVariations = {100, 150, 200, 50};

    std::vector<int> barPadding = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Get timestamp
    time_t rawtime;
    struct tm * timeinfo;
    char timeStr[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(timeStr,sizeof(timeStr),"%Y-%m-%d_%H-%M-%S", timeinfo);

    for (size_t k = 0; k < numAgentVariations.size(); ++k){
      size_t numAgents = numAgentVariations[k];
      // If using softmax, specifying number of agents to not learn doesn't matter
      if (adaptiveLearning == 2){
        for (size_t j = 0; j < numTrials; ++j){
            runMultiTrials(timeStr, numAgents, 0, j, barPadding, adaptiveLearning, maxEpoch);
        }
      }
      else{
        for (size_t i = 0; i < 20; ++i){
            size_t numDisabled = i*10; 
            for (size_t j = 0; j < numTrials; ++j){
                if (numAgents > numDisabled){
                    runMultiTrials(timeStr, numAgents, numDisabled, j, barPadding, adaptiveLearning, maxEpoch);
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
}

