#ifndef MULTI_NIGHT_BAR_Q_H_
#define MULTI_NIGHT_BAR_Q_H_

#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Eigen>
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>

#include "Learning/QLearner.h"
#include "Bar.h"

class MultiNightBarQ{
  public:
    MultiNightBarQ(size_t nNights, size_t cap, size_t nAgents, std::string evalFunc, 
                   double lr, double discountFactor, double probRandom, double maxReward,
                   size_t nAgentsDisabled, bool dLearning);

    MultiNightBarQ(size_t nNights, size_t cap, std::vector<int> barOccupancyPad, size_t nAgents, std::string evalFunc, 
                   double lr, double discountFactor, double probRandom, double maxReward,
                   size_t nAgentsDisabled, bool dLearning);

    ~MultiNightBarQ();
    
    /**
     * InitialiseEpoch()
     * 
     * \brief Clears and initiate Bar objects for each night
     */
    void initialiseEpoch();

    /**
     * SimulateEpoch()
     * 
     * \brief Runs an epoch of Q-Learning, update agent Q values if train is set to true
     *        Returns G for the epoch simulated
     *
     */
    double simulateEpoch(bool train = true);

    /**
     * computeG()
     * 
     * \brief Given a vector of occupants per bar, computes the G value
     */
    double computeG(std::vector<size_t> occupancy);

    /**
     * train()
     * 
     * \brief Trains the Q agents for numEpochs
     * \note NOT IMPLEMENTED YET, use simulateEpoch instead
     */
    void train(size_t numEpochs);

    /**
     * computeFinalScore()
     * 
     * \brief Determines Final Global Reward with all agents using their best action
     */
    double computeFinalScore();

     /**
     * computeFinalScore()
     * 
     * \brief Determines Final Global Reward with all agents using their best action
    *         And writes out the Q table and actions of each of the agents
     */
    double computeFinalScoreOutput(char* qTablePath, char* actionPath);

    /**
     * outputActions()
     * 
     * \brief Outputs the agent's action in the following format 
     *        night number. number of agents attending that night, enjoyment 
     */   
    void outputActions(char* B, std::vector<size_t> barOccupancy);

    /**
     * outputQTables()
     * 
     * \brief Writes the Agents' Q Tables out to a file
     * \note Leverages the fact that Agents are single state,
     *       otherwise states in output file will get smooshed
     */
    void outputQTables(char* A);

    /**
     * outputParameters()
     * 
     * \brief Writes the parameters of the run out to a file
     */
    void outputParameters(char* fname, size_t numEpochs);

    /**
     * outputAgentActions()
     * 
     * \brief Writes the best actions of the agent out to a file
     */
    void outputAgentActions(char* fname);    



  private:
    size_t numNights;
    size_t capacity;
    std::string evaluationFunction;
    size_t numAgents;
    size_t numAgentsDisabled;
    
    vector<Bar> barNights;
    vector<QLearner *> agents;

    bool outputEvals;
    bool outputActs;

    bool useD;
    bool dynamicLearning;

    double prevG; /// the previous global reward
    
    std::ofstream evalFile;
    std::ofstream actFile;
    std::ofstream barFile;
    std::ofstream QTableFile;

    double learningRate; /// learning rate (alpha)
    double discountFactor; /// discount factor (gamma)
    double epsilon; /// chance of selecting random action
    
    std::vector<int> barOccupancyPadding; // for each bar, how much should the occupancy be padded
};

#endif // MULTI_NIGHT_BAR_Q_H_
