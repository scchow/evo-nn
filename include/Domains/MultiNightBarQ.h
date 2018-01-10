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
                   size_t nAgentsDisabled, int adaptiveLearning, size_t mEpoch, double temperature);

    MultiNightBarQ(size_t nNights, size_t cap, std::vector<int> barOccupancyPad, size_t nAgents, std::string evalFunc, 
                   double lr, double discountFactor, double probRandom, double maxReward,
                   size_t nAgentsDisabled, int adaptiveLearning, size_t mEpoch, double temperature);

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
     * \brief Runs an epoch of Q-Learning, update agent Q values 
     *
     */
    double simulateEpoch(size_t epochNumber);

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

    void outputNumLearning(char* fname, size_t numEpochs);

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

    double temperature(size_t epochNumber);

  private:
    size_t numNights;
    size_t capacity;
    std::string evaluationFunction;
    size_t numAgents;
    size_t numAgentsDisabled;
    size_t numAgentsLearning;
    
    vector<Bar> barNights;
    vector<QLearner *> agents;

    bool outputEvals;
    bool outputActs;

    bool useD;

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

    int adaptive;

    double prevG; /// the previous global reward
    std::vector<double> prevD; /// the previous D rewards for each agent

    
    std::ofstream evalFile;
    std::ofstream actFile;
    std::ofstream barFile;
    std::ofstream QTableFile;
    std::ofstream numLearningFile;

    double discountFactor; /// discount factor (gamma)
    double learningRate; /// learning rate (alpha)
    double epsilon; /// chance of selecting random action

    double temp;
    size_t maxEpoch; /// the final epoch number simulated (used for temperature calculations)
    
    std::vector<int> barOccupancyPadding; // for each bar, how much should the occupancy be padded

    std::random_device rd; /// Seed Generator
    std::mt19937_64 generator{rd()}; /// generator initialized with seed from rd
    std::uniform_real_distribution<> distReal{0.0, 1.0}; /// Random number distribution from 0 to 1
};

#endif // MULTI_NIGHT_BAR_Q_H_
