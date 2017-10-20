#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Eigen>

#include "Domains/MultiNightBarQ.h"

using std::vector ;
using std::string ;
using namespace Eigen ;

int main(){
  std::cout << "Testing MultiNightBarQ class in MultiNightBarQ.h\n" ;
  
  size_t capacity = 10 ;
  size_t numNights = 10 ;
  size_t numAgents = 100 ;
  size_t nAgentsDisabled = 0;
  string evalFunc = "D" ;

  double learningRate = 0.1;
  double discount = 0.9;
  double epsilon = 0.01;
  double maxReward = 100;

  size_t nEps = 200;

  std::cout << "This program will use Q-learning to train " << numAgents << "-agent team over " << nEps << " learning epochs\n" ;
  std::cout << nAgentsDisabled << " Agents will be not be learning\n";
  std::cout << "Agent Q-Learning parameters:\n" ;
  std::cout << "  Learning Rate: " << learningRate << "\n" ;
  std::cout << "  Discount: " << discount << "\n" ;
  std::cout << "  Epsilon: " << epsilon << "\n" ;
  std::cout << "  Evaluation function: " << evalFunc << "\n" ;
  std::cout << "Environment parameters:\n" ;
  std::cout << "  Bar capacity: " << capacity << "\n" ;
  std::cout << "  Number of nights: " << numNights << "\n" ;

  
  // int trialNum ;
  // std::cout << "Please enter trial number [NOTE: no checks enabled to prevent overwriting existing files, user must make sure trial number is unique]: " ;
  // std::cin >> trialNum ;
  
  MultiNightBarQ trainDomain(numNights, capacity, numAgents, evalFunc, 
                               learningRate, discount, epsilon, maxReward, nAgentsDisabled) ;
  
  // int buffSize = 100 ;
  // char fileDir[buffSize] ;
  // // sprintf(fileDir,"Results/MultiNightBar/%d_nights/%d_epochs/%d_agents/%d_disabled/%s/trial_%d",(int)numNights,(int)nEps, (int) agents, (int)nAgentsDisabled, evalFunc.c_str(),trialNum) ;
  // sprintf(fileDir,"Results/MultiNightBar/%d_nights/%d_epochs/%d_agents/0_disabled/%s/trial_%d",(int)numNights,(int)nEps, (int) numAgents, evalFunc.c_str(),trialNum) ;
  // char mkdir[buffSize] ;
  // sprintf(mkdir,"mkdir -p %s",fileDir) ;
  // system(mkdir) ;
  
  // std::cout << "\nWriting log files to: " << fileDir << "\n\n" ;
  
  // char eFile[buffSize] ;
  // sprintf(eFile,"%s/results.txt",fileDir) ;
  // char aFile[buffSize] ;
  // sprintf(aFile,"%s/actions.txt",fileDir) ;
  // char cFile[buffSize] ;
  // sprintf(cFile,"%s/capacities.txt",fileDir) ;
  // char readmeFile[buffSize];
  // sprintf(readmeFile, "%s/readme.txt",fileDir);
  // char pveFile[buffSize];
  // sprintf(pveFile, "%s/performance_vs_epoch.csv",fileDir);


  // trainDomain.OutputPerformance(eFile) ;
  // trainDomain.OutputParameters(readmeFile);

  // std::cout << "Beginning Training";
  double G;

  for (size_t n = 0; n < nEps; n++){
    std::cout << "Episode " << n << "...\n" ;
    trainDomain.initialiseEpoch();
    // std::cout << "Initiailized epoch";
    G = trainDomain.simulateEpoch(true);
    std::cout << "Score " << G << std::endl; 
    std::cout << "Q[0][numActions-1] value: " << trainDomain.getQ00() <<std::endl;
 
  }
  std::cout << "Final Score" << std::endl;
  G = trainDomain.computeFinalScore();
  std::cout << "Score " << G << std::endl; 
  // char NNFile[buffSize] ;
  // sprintf(NNFile,"%s/NNs.txt",fileDir) ;
  
  // std::cout << "\nWriting final control policies to file...\n" ;
  
  // trainDomain.OutputControlPolicies(NNFile) ;
  
  // int isTest ;
  // std::cout << "Please enter [0] to end program, [1] to test stored NN policies in new environment: " ;
  // std::cin >> isTest ;
  // if (isTest < 0 || isTest > 1){
  //   std::cout << "Input is out of range. Exiting program.\n" ;
  //   isTest = 0 ;
  // } 
  
  // if (isTest == 1){
    
  //   capacity = 10 ;
  //   numNights = 10 ;
  //   std::cout << "Test parameters:\n" ;
  //   std::cout << "  Bar capacity: " << capacity << "\n" ;
  //   std::cout << "  Number of nights: " << numNights << "\n" ;
    
  //   std::cout << "Testing stored control policies on new world...\n" ;
    
  //   MultiNightBar testDomain(numNights, capacity, nPop, evalFunc, agents, nAgentsDisabled) ;

  //   char eeFile[buffSize] ;
  //   sprintf(eeFile,"%s/results_test.txt",fileDir) ;
  //   char aaFile[buffSize] ;
  //   sprintf(aaFile,"%s/actions_test.txt",fileDir) ;
  //   char ccFile[buffSize] ;
  //   sprintf(ccFile,"%s/bars_test.txt",fileDir) ;
    
  //   testDomain.ExecutePolicies(NNFile, aaFile, ccFile, eeFile, nInputs, nOutputs, nHidden) ;
  // }
  
  std::cout << "Test complete!\n" ;
  
  return 0 ;
}
