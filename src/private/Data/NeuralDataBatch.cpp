//
// Created by wysep on 11/29/2024.
//

#include "Data/NeuralDataBatch.h"


NeuralDataBatch::NeuralDataBatch() = default;

NeuralDataBatch::NeuralDataBatch(std::vector<NeuralDataObject> neuralDataObjects)
{
    this->neuralDataObjects = neuralDataObjects;
}
