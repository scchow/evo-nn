#include "MultiNightBarQ.h"

MultiNightBarQ::MultiNightBarQ(size_t nNights, size_t cap, size_t nAgents, std::string evalFunc, 
                               double lr, double discount, double probRandom, double maxReward, size_t nAgentsDisabled): 
                               numNights(nNights), capacity(cap), numAgents(nAgents), numAgentsDisabled(nAgentsDisabled){

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

    std::cout << "\nSimulating Epoch Results:" << std::endl;
    for (size_t i = 0; i < barNights.size(); ++i){ // compute reward for each night and sum
        std::cout << "Night number: " << i << ", attendance: " << barOccupancy[i] << ", enjoyment: " << barNights[i].GetReward(barOccupancy[i]) << "\n" ;
    }

    // Compute G
    double G = MultiNightBarQ::computeG(barOccupancy);

    if (train){
        if (useD){
            //Compute D
            std::vector<double> D_vec;

            for (size_t i = 0; i < numAgents; ++i){
                int agentAction = agents[i]->getCurrentAction();

                // Remove agent from bar
                barOccupancy[agentAction]--;

                double G_hat = MultiNightBarQ::computeG(barOccupancy);

                D_vec.push_back(G-G_hat);

                // Add agent back into bar
                barOccupancy[agentAction]++;
            }

            for (size_t i = 0; i < numAgents; ++i){
                // Pass D as reward and remain at state 0
                agents[i]->updateQ(D_vec[i], 0);
            }

        }

        // otherwise not using D, pass in G instead
        else{

            for (size_t i = 0; i < numAgents; ++i){
                // Pass G as reward and remain at state 0
                agents[i]->updateQ(G, 0);
            }

        }
    }


    return G;
}

double MultiNightBarQ::computeFinalScore(){
    // Get actions from each agent and
    // keep track of how many went to each night
    vector<size_t> barOccupancy(numNights, 0);

    for (size_t i = 0; i < numAgents; ++i){
        size_t action = agents[i]->getBestAction();
        barOccupancy[action]++;
    }
    std::cout << "\nTesting Best Action Results:" << std::endl;
    for (size_t i = 0; i < barNights.size(); ++i){ // compute reward for each night and sum
        std::cout << "Night number: " << i << ", attendance: " << barOccupancy[i] << ", enjoyment: " << barNights[i].GetReward(barOccupancy[i]) << "\n";
    }

    // Compute G
    double G = MultiNightBarQ::computeG(barOccupancy);

    return G;
}


double MultiNightBarQ::computeFinalScoreOutput(char* qTablePath, char* actionPath){
    // Get actions from each agent and
    // keep track of how many went to each night
    vector<size_t> barOccupancy(numNights, 0);

    for (size_t i = 0; i < numAgents; ++i){
        size_t action = agents[i]->getBestAction();
        barOccupancy[action]++;
    }
    std::cout << "\nTesting Best Action Results:" << std::endl;
    for (size_t i = 0; i < barNights.size(); ++i){ // compute reward for each night and sum
        std::cout << "Night number: " << i << ", attendance: " << barOccupancy[i] << ", enjoyment: " << barNights[i].GetReward(barOccupancy[i]) << "\n";
    }

    // Compute G
    double G = MultiNightBarQ::computeG(barOccupancy);

    MultiNightBarQ::outputQTables(qTablePath);
    MultiNightBarQ::outputActions(actionPath, barOccupancy);

    return G;
}

// // Wrapper for writing epoch evaluations to specified files
// void MultiNightBarQ::OutputPerformance(char* A){
//     // Filename to write to stored in A
//     std::stringstream fileName;
//     fileName << A;
//     if (evalFile.is_open()){
//         evalFile.close();
//     }
//     evalFile.open(fileName.str().c_str(),std::ios::app);

//     outputEvals = true;
// }

// Wrapper for writing agent actions to specified files
void MultiNightBarQ::outputActions(char* B, std::vector<size_t> barOccupancy){
    // Filename to write bar configurations to stored in B
    std::stringstream barStream;
    barStream << B;
    if (barFile.is_open()){
        barFile.close();
    }
    barFile.open(barStream.str().c_str(),std::ios::app);
    for (size_t i = 0; i < barOccupancy.size(); ++i){ // compute reward for each night and sum
        barFile << i << ", "; // Night Number
        barFile << barOccupancy[i] << ", " ; // Occupancy
        barFile << barNights[i].GetReward(barOccupancy[i]) << "\n"; //Enjoyment
    }
}

// Wrapper for writing final control policies to specified file
void MultiNightBarQ::outputQTables(char* A){
    for (size_t i = 0; i < numAgents; i++)
        agents[i]->outputQTable(A);
}

void MultiNightBarQ::outputParameters(char* fname, size_t numEpochs){
    std::stringstream fnameStream;
    fnameStream << fname;

    std::ofstream paramFile;

    paramFile.open(fnameStream.str().c_str(), std::ios::app);
    paramFile << "Nights: "<< numNights << "\n";
    paramFile << "Capacity: " << capacity << "\n";
    paramFile << "Number Agents: "<< numAgents << "\n";
    paramFile << "Number Agents Disabled: " << numAgentsDisabled << "\n";
    paramFile << "Evaluation Function: " << evaluationFunction << "\n";
    paramFile << "Agent Q-Learning parameters:\n";
    paramFile << "  Learning Rate: " << learningRate << "\n";
    paramFile << "  Discount: " << discountFactor << "\n";
    paramFile << "  Epsilon: " << epsilon << "\n";
    paramFile << "Epochs: " << numEpochs << "\n";

    paramFile.close();
}
