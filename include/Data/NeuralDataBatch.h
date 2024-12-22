//
// Created by wysep on 11/29/2024.
//

#ifndef NEURALDATABATCH_H
#define NEURALDATABATCH_H

#include "NeuralDataFile.h"
#include "Neural/NeuralNetworkResult.h"

class NeuralDataBatch
{
private:
    std::vector<std::shared_ptr<NeuralDataObject>> neuralDataObjects;
    long long iterationIDStart;

public:
    NeuralDataBatch();
    NeuralDataBatch(std::vector<std::shared_ptr<NeuralDataObject>> neuralDataObjects , long long iterationStartID);

    std::vector<std::shared_ptr<NeuralDataObject>> GetNeuralDataObjects();
    long long GetIterationIDStart();
    std::shared_ptr<NeuralNetworkGroupResult> groupResultRef;

};

#endif //NEURALDATABATCH_H
