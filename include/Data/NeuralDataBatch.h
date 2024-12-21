//
// Created by wysep on 11/29/2024.
//

#ifndef NEURALDATABATCH_H
#define NEURALDATABATCH_H

#include "NeuralDataFile.h"

struct NeuralDataBatch
{
private:
    std::vector<NeuralDataObject> neuralDataObjects;

public:
    NeuralDataBatch();
    NeuralDataBatch(std::vector<std::shared_ptr<NeuralDataObject>> neuralDataObjects);
};

#endif //NEURALDATABATCH_H
