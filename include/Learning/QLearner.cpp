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
    // If stop learning, lock the agent to a specific action
    if (flag == false){
        // set the agent to perform the best action it has so far (random if no training at start)
        currAction = QLearner::getBestAction();
    }
    train = flag;
}

size_t QLearner::getAction(){
    // If agent is not learning, just return set action
    if (train==false){
        return currAction;
    }
    double rand = distReal(generator);
    size_t action = 0;
    // std::cout << "Rand num:" << rand << std::endl;
    // std::cout << "currentState " << currState << "\n";
    // with probability epsilon, choose a random action
    if (rand < epsilon){
        int randAction = distInt(generator);
        // std::cout << "Rand action:" << randAction << std::endl;

        action = randAction;
        // std::cout << "new action "<< action << "\n";
    }
    // otherwise go greedy and choose best action
    else{
        // std::cout << "going greedy, rand " << rand << "\n";
        std::vector<double> maxIndices = getMaxIndices(Q[currState]);
        if (maxIndices.size() == 1){
            action = maxIndices[0];
        }
        else{
            std::uniform_int_distribution<> dist(0, maxIndices.size()-1);
            int randInt = dist(generator);
            action = maxIndices[randInt];
            // std::cout << "Multiple best values, picking index: " << action << std::endl;
        }
        // std::cout << "max greedy action "<< action << "\n";
    }

    currAction = action;
    return action;
}

size_t QLearner::getBestAction(){
    // If agent is not learning, just return set action
    if (train==false){
        return currAction;
    }
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
        std::uniform_int_distribution<> dist(0, maxIndices.size()-1);
        int randInt = dist(generator);
        action = maxIndices[randInt];
        // std::cout << "Multiple best values, picking index: " << action << std::endl;
    }
    return action;
}

size_t QLearner::getCurrentAction(){
    return currAction;
}

void QLearner::updateQ(double reward, size_t nextState){
    
    if (train){
        double maxValueNextState = *std::max_element( Q[currState].begin(), Q[currState].end() );
        
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

void QLearner::outputQTable(char * A){
    std::stringstream fileName;
    fileName << A;
    std::ofstream QTableFile;
    QTableFile.open(fileName.str().c_str(),std::ios::app);

    for (size_t i = 0; i < Q.size(); ++i){
        for (size_t j = 0; j < Q[i].size(); ++j){
            QTableFile << Q[i][j] << ",";
        }
        QTableFile << "\n";
    }

    QTableFile.close();
}
