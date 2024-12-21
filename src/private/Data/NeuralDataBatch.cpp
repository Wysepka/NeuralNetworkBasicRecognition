//
// Created by wysep on 11/29/2024.
//

#include "Data/NeuralDataBatch.h"


NeuralDataBatch::NeuralDataBatch() = default;

std::vector<std::shared_ptr<NeuralDataObject>> NeuralDataBatch::GetNeuralDataObjects()
{
    return neuralDataObjects;
}

long long NeuralDataBatch::GetIterationIDStart()
{
    return iterationIDStart;
}

NeuralDataBatch::NeuralDataBatch(std::vector<std::shared_ptr<NeuralDataObject>> neuralDataObjects , long long iterationIDStart)
{
    this->neuralDataObjects = neuralDataObjects;
    this->iterationIDStart = iterationIDStart;
}
