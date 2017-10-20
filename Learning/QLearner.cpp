#include "QLearner.h"

QLearner::QLearner(double lr, double discount, double probRandom, double maxReward, size_t nStates, size_t nActions, size_t initState): 
                   learningRate(lr), discountFactor(discount), epsilon(probRandom), numStates(nStates), numActions(nActions){

    for (size_t i = 0; i < numStates; ++i){
        std::vector<double> vec(numActions, maxReward);
        Q.push_back(vec);
    }

    //set initial state
    currState = initState;

    train = true;

    distInt = std::uniform_int_distribution<>(0, nActions);

    // let currAction be NULL since no actions have been performed yet
    // currAction = NULL;
    // std::cout << "Q Learner Initialized";
}

QLearner::~QLearner(){
    return;
}


void QLearner::setLearningFlag(bool flag){
    train = flag;
}

size_t QLearner::getAction(){
    // TODO: optimize by turning uniform distribution into data memeber
    
    // std::mt19937 gen{rd()};
    // std::uniform_real_distribution<> dis(0.0, 1.0);
    double rand = distReal(generator);
    size_t action = 0;
    // std::cout << "Rand num:" << rand << std::endl;
    // std::cout << "currentState " << currState << "\n";
    // with probability epsilon, choose a random action
    if (rand < epsilon){
        int randAction = distInt(generator);
        std::cout << "Rand action:" << randAction << std::endl;

        action = randAction;
        // std::cout << "new action "<< action << "\n";
    }
    // otherwise go greedy and choose best action
    else{
        // std::cout << "going greedy, rand " << rand << "\n";
        std::vector<double> maxIndices = getMaxIndices(Q[currState]);
        // for (size_t i = 0; i < maxIndices.size(); ++i){
        //     std::cout << maxIndices[i];
        // }
        // std::cout << "\n";
        if (maxIndices.size() == 1){
            action = maxIndices[0];
        }
        else{
            std::uniform_int_distribution<> dist(0, maxIndices.size());
            int randInt = dist(generator);
            action = randInt;
            // std::cout << "Multiple best values, picking index: " << action << std::endl;
        }
        // std::cout << "max greedy action "<< action << "\n";
    }

    currAction = action;
    return action;
}

size_t QLearner::getBestAction(){
    size_t action = 0;
    std::vector<double> maxIndices = getMaxIndices(Q[currState]);
    // for (size_t i = 0; i < maxIndices.size(); ++i){
    //     std::cout << maxIndices[i];
    // }
    // std::cout << "\n";
    if (maxIndices.size() == 1){
        action = maxIndices[0];
    }
    else{
        std::uniform_int_distribution<> dist(0, maxIndices.size());
        int randInt = dist(generator);
        action = randInt;
        // std::cout << "Multiple best values, picking index: " << action << std::endl;
    }
    return action;
}

void QLearner::updateQ(double reward, size_t nextState){
    
    if (train){
        double maxValueNextState = std::distance(Q[nextState].begin(), std::max_element(Q[nextState].begin(), Q[nextState].end()));
        
        // Update Q-value table
        Q[currState][currAction] = 
            Q[currState][currAction] + 
            learningRate * (reward + (discountFactor * maxValueNextState) - Q[currState][currAction]);
    }
    // Update state to next state
    currState = nextState;
    // let actions be NULL again, since action no longer associated with currState
    // this will ensure getAction() is called before updateQ()
    // currAction = NULL;
}


