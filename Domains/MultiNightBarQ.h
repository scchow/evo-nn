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
                   double lr, double discountFactor, double probRandom, double maxReward);
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
     */
    void train(size_t numEpochs);


  private:
    size_t numNights;
    size_t capacity;
    std::string evaluationFunction;
    size_t numAgents;
    
    vector<Bar> barNights;
    vector<QLearner *> agents;

    bool outputEvals;
    bool outputActs;

    bool useD;
    
    std::ofstream evalFile;
    std::ofstream actFile;
    std::ofstream barFile;
    std::ofstream NNFile;

    double learningRate; /// learning rate (alpha)
    double discountFactor; /// discount factor (gamma)
    double epsilon; /// chance of selecting random action
};

#endif // MULTI_NIGHT_BAR_Q_H_
