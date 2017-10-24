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

using namespace easymath;

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


        /**
         * setLearningFlag()
         *
         * \brief starts or stops the agent from learning 
         */
        void setLearningFlag(bool flag);

        /**
         * getBestAction
         *
         * \brief Gets the next action the agent will take
         */
        size_t getBestAction();

        /**
         * getCurrentAction
         *
         * \brief Gets the action the agent has taken in current state
         * \note Must be called after getAction() for current state
         */
        size_t getCurrentAction();

        /**
         * outputQTable
         *
         * \brief Writes the Agent's Q table out to file
         */
        void outputQTable(char * A);

        bool isLearning(){
            return train;
        }


    private:
        double learningRate; /// learning rate (alpha)
        double discountFactor; /// discount factor (gamma)
        double epsilon; /// chance of selecting random action
        size_t numStates; /// number of states
        size_t numActions; /// number of actions
        std::vector<std::vector<double>> Q; /// Q state-action 
        size_t currState; /// current state represented by index
        size_t currAction; /// current action represented by index
        bool train; /// Determine if this agent will learn (update Q value)
        std::random_device rd; /// Seed Generator
        std::mt19937_64 generator{rd()}; /// generator initialized with seed from rd
        std::uniform_real_distribution<> distReal{0.0, 1.0}; /// Random number distribution from 0 to 1
        std::uniform_int_distribution<> distInt{0, 2}; /// Random int distribution (to be overwritten upon actual init)
};

#endif // QLEARNER_H_