#ifndef QLEARNING_H_
#define QLEARNING_H_

#include <stdio.h>
#include <iostream>
#include <float.h>
#include <math.h>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Eigen>
#include "Utilities/Utilities.h"


class QLearning{
    public:
        /**
         * QLearning Constructor
         */
        QLearning(size_t numAgents, float learningRate, float discountFactor, float probRandom);

        ~QLearning();

    private:
        float lr; /// learning rate
        float discount; /// discount factor
        float epsilon; /// chance of selecting random action
        std::vector<float> v;

}

#endif // QLEARNING_H_