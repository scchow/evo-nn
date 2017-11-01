#include "MultiNightBarQ.h"

MultiNightBarQ::MultiNightBarQ(size_t nNights, size_t cap, size_t nAgents, std::string evalFunc, 
                               double lr, double discount, double probRandom, double maxReward, 
                               size_t nAgentsDisabled, int adaptiveLearning, size_t mEpoch): 
                               numNights(nNights), capacity(cap), numAgents(nAgents), 
                               numAgentsDisabled(nAgentsDisabled), adaptive(adaptiveLearning),
                               maxEpoch(mEpoch){

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
        
    }

    for (size_t i = 0; i < nNights; ++i){
        barOccupancyPadding.push_back(0);
    }

}

MultiNightBarQ::MultiNightBarQ(size_t nNights, size_t cap, std::vector<int> barOccupancyPad, size_t nAgents, std::string evalFunc, 
                               double lr, double discount, double probRandom, double maxReward, size_t nAgentsDisabled, 
                               int adaptiveLearning, size_t mEpoch): 
                               numNights(nNights), capacity(cap), numAgents(nAgents), 
                               numAgentsDisabled(nAgentsDisabled), adaptive(adaptiveLearning),
                               maxEpoch(mEpoch), barOccupancyPadding(barOccupancyPad){

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
        barNights.push_back(Bar(capacity, barOccupancyPadding[i]));
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

double MultiNightBarQ::simulateEpoch(size_t epochNumber){
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

    // Choose agents to disable
    // Note, disable agents after updating so that they get updated first
    std::vector<bool> newLearningStates(numAgents, 0);
    if (adaptive){
        double deltaG = G - prevG;
        std::vector<double> impacts;

        // get impact of each agent
        for (size_t i = 0; i < numAgents; ++i){
            impacts.push_back(agents[i]->computeImpact(deltaG));
        }

        // list of agents from least to most impactful
        std::vector<size_t> sortedIndices = sortIndices(impacts);
        // size_t numLearningAgents = numAgents - numAgentsDisabled;

        if (adaptive == 1){
            // select the most impactful quarter of agents to continue training
            for (size_t i = numAgentsDisabled; i < numAgents; ++i){
                newLearningStates[sortedIndices[i]] = true;
            }
        }

        else if (adaptive == 2){
            int numLearning = 0;
            for (size_t i = 0; i < numAgents; ++i){
                double prob = 1- std::exp(-1 * impacts[i] / MultiNightBarQ::temperature(epochNumber));
                // std::cout << "prob = " << prob << std::endl;
                double rand = distReal(generator);
                if (rand < prob){
                    newLearningStates[i] = true;
                    numLearning += 1;
                }
            }
            std::cout << "number agents learning = " << numLearning << std::endl;
        }

        std::cout << "Impact Vector: ";
        for (size_t i = 0; i < impacts.size(); ++i){ // compute reward for each night and sum
            std::cout << impacts[i] << "," ;
        }
        std::cout << "\n";


    }

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

    // disable agents here
    if (adaptive){
        for (size_t i = 0; i < numAgents; ++i){
            // Pass G as reward and remain at state 0
            agents[i]->setLearningFlag(newLearningStates[i]);
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
        barFile << barNights[i].GetPadding() << ", "; // Padding
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

void MultiNightBarQ::outputAgentActions(char* fname){
    //Write the file path to a stream
    std::stringstream fnameStream;
    fnameStream << fname;

    //use stream to open file
    std::ofstream actionFile;
    actionFile.open(fnameStream.str().c_str(), std::ios::app);

    // Write the best actions to a file
    for (size_t i = 0; i < agents.size(); ++i){
        actionFile << i << ", " << agents[i]->isLearning() << ", " << agents[i]->getBestAction() << "\n"; 
    }

    actionFile.close();
}

double MultiNightBarQ::temperature(size_t epochNumber){
    // try a linear temperature function for now
    // std::cout << "temperature = " << 100 * epochNumber/maxEpoch << std::endl;
    return epochNumber/maxEpoch;

}