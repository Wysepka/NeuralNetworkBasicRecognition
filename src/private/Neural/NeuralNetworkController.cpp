//
// Created by Marcin on 11/15/2024.
//

#include "Neural/NeuralNetworkController.h"

void NeuralNetworkController::Initialize(std::shared_ptr<NeuralDataFile> neuralDataFile)
{
    network = std::make_shared<NeuralNetwork>();
    this->neuralDataFile = neuralDataFile;
    std::vector<NeuralNetworkHiddenConfig> hiddenConfig;
    hiddenConfig.push_back(NeuralNetworkHiddenConfig(100));

    auto sigmoidActivation = std::make_shared<Sigmoid>();
    auto crossEntropyCost = std::make_shared<CrossEntropy>();

    NeuralNetworkConfig neuralConfig(neuralDataFile->GetNeuralDataObjects()[0]->GetFlatObjectPixelsArray_Normalized().size() , hiddenConfig , 10 , sigmoidActivation , crossEntropyCost);

    network->SetConfig(neuralConfig);
}

void NeuralNetworkController::Run(bool& shouldBeRunning)
{
    if(!shouldBeRunning)
    {
        Dispose();
        return;
    }

    if(isCurrentlyRunning)
    {
        return;
    }



    network->RunNetwork(neuralDataFile);

    *isCurrentlyRunning = true;
}

void NeuralNetworkController::Dispose()
{

}
