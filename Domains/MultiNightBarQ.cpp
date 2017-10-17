#include "MultiNightBarQ.h"

MultiNightBarQ::MultiNightBarQ(size_t nNights, size_t cap, size_t nAgents, std::string evalFunc, 
                               double lr, double discount, double probRandom, double maxReward, size_t nAgentsDisabled): 
                               numNights(nNights), capacity(cap), 
                               numAgents(nAgents){

    size_t numStates = 1; // single state problem
    size_t numActions = nNights; // each agent can choose which night to go on
    size_t initState = 0; // all agents start with the only state they can be on

    // Create QLearning Agents for each of the agents
    for (size_t i = 0; i < numAgents; ++i){
        QLearner* newAgent = new QLearner(lr, discount, probRandom, maxReward, 
                                        numStates, numActions, initState);
        if (i < nAgentsDisabled){
            newAgent->setLearningFlag(false);
        }
        agents.push_back(newAgent);
    }

    // Determine Evaluation Function to use
    if (evalFunc.compare("D") == 0){
        useD = true;
    }
    else if (evalFunc.compare("G") == 0){
        useD = false;
    }
    else{
        std::cout << "ERROR: Unknown evaluation function type [" << evalFunc << "], setting to global evaluation!\n";
        useD = false;
    }

}

MultiNightBarQ::~MultiNightBarQ(){
    for (size_t i = 0; i < numAgents; ++i){
        delete agents[i];
        agents[i] = nullptr;
    }
}

void MultiNightBarQ::initialiseEpoch(){
    // Initialise each night as a separate Bar object
    barNights.clear();
    for (size_t i = 0; i < numNights; ++i){
        barNights.push_back(Bar(capacity));
    }
}

double MultiNightBarQ::computeG(std::vector<size_t> occupancy){
    double G = 0.0;
    for (size_t i = 0; i < barNights.size(); ++i){ // compute reward for each night and sum
      G += barNights[i].GetReward(occupancy[i]);
      // std::cout << "Night number: " << i << ", attendance: " << occupancy[i] << ", enjoyment: " << barNights[i].GetReward(barOccupancy[i]) << "\n" ;
    }
    return G;
}

double MultiNightBarQ::simulateEpoch(bool train){
    // std::cout << "began simulateEpoch " << "\n";
    // Get actions from each agent and
    // keep track of how many went to each night
    vector<size_t> barOccupancy(numNights, 0);

    for (size_t i = 0; i < numAgents; ++i){
        // std::cout << "getting Action of agent "<< i << "\n";
        size_t action = agents[i]->getAction();
        // std::cout << "got action " << "\n";
        barOccupancy[action]++;
    }

    for (size_t i = 0; i < barNights.size(); ++i){ // compute reward for each night and sum
      std::cout << "Night number: " << i << ", attendance: " << barOccupancy[i] << ", enjoyment: " << barNights[i].GetReward(barOccupancy[i]) << "\n" ;
    }
    // Compute G
    double G = MultiNightBarQ::computeG(barOccupancy);

    if (useD){
        //Compute D
        std::vector<double> D_vec;
        for (size_t i = 0; i < numAgents; ++i){
            int agentAction = agents[i]->getAction();

            // Remove agent from bar
            barOccupancy[agentAction]--;

            double G_hat = MultiNightBarQ::computeG(barOccupancy);

            D_vec.push_back(G-G_hat);

            // Add agent back into bar
            barOccupancy[agentAction]++;
        }
        // Update Q-Values
        if (train){
            for (size_t i = 0; i < numAgents; ++i){
                // Pass D as reward and remain at state 0
                agents[i]->updateQ(D_vec[i], 0);
            }
        }
    }

    // otherwise not using D, pass in G instead
    else{
        if (train){
            for (size_t i = 0; i < numAgents; ++i){
                // Pass G as reward and remain at state 0
                agents[i]->updateQ(G, 0);
            }
        }
    }

    return G;
}

double MultiNightBarQ::computeFinalScore(){
    // std::cout << "began simulateEpoch " << "\n";
    // Get actions from each agent and
    // keep track of how many went to each night
    vector<size_t> barOccupancy(numNights, 0);

    for (size_t i = 0; i < numAgents; ++i){
        size_t action = agents[i]->getBestAction();
        barOccupancy[action]++;
    }

    for (size_t i = 0; i < barNights.size(); ++i){ // compute reward for each night and sum
      std::cout << "Night number: " << i << ", attendance: " << barOccupancy[i] << ", enjoyment: " << barNights[i].GetReward(barOccupancy[i]) << "\n" ;
    }

    // Compute G
    double G = MultiNightBarQ::computeG(barOccupancy);

    return G;
}
void MultiNightBarQ::train(size_t numEpochs){
    for (size_t epochInd = 0; epochInd < numEpochs; ++epochInd){
        MultiNightBarQ::initialiseEpoch();
        MultiNightBarQ::simulateEpoch(true);
    }

}


