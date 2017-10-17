#ifndef QLEARNER_H_
#define QLEARNER_H_

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Eigen>
#include "Utilities/Utilities.h"


class QLearner{
    public:
        /**
         * QLearning Constructor
         */
        QLearner(double lr, double discount, double probRandom, double maxReward, 
                 size_t nStates, size_t nActions, size_t initState);

        /**
         * QLearning Destructor
         */
        ~QLearner();

        /**
         * getAction
         *
         * \brief Gets the next action the agent will take
         */
        size_t getAction();

        /**
         * updateQ
         *
         * \brief Updates the Q value after an action is taken and Updates the State
         */
        void updateQ(double reward, size_t nextState);

    private:
        double learningRate; /// learning rate (alpha)
        double discountFactor; /// discount factor (gamma)
        double epsilon; /// chance of selecting random action
        size_t numStates; /// number of states
        size_t numActions; /// number of actions
        std::vector<std::vector<double>> Q; /// Q state-action 
        size_t currState; /// current state represented by index
        size_t currAction; /// current action represented by index

};

#endif // QLEARNER_H_