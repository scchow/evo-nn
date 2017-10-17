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
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double rand = dis(gen);
    size_t action = 0;

    // std::cout << "currentState " << currState << "\n";
    // with probability epsilon, choose a random action
    if (epsilon < rand){
        action = std::rand() % numActions;
        // std::cout << "new action "<< action << "\n";
    }
    // otherwise go greedy and choose best action
    else{
        // std::cout << "going greedy, rand " << rand << "\n";
        action = std::distance(Q[currState].begin(), std::max_element(Q[currState].begin(), Q[currState].end()));
        // std::cout << "new action "<< action << "\n";
    }

    currAction = action;
    return action;
}

size_t QLearner::getBestAction(){
    size_t action = std::distance(Q[currState].begin(), std::max_element(Q[currState].begin(), Q[currState].end()));
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


