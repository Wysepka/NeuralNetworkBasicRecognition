//
// Created by Marcin on 11/15/2024.
//

#ifndef NEURALNETWORKUTILITY_H
#define NEURALNETWORKUTILITY_H
#include <vector>
#include <Data/NeuralDataFile.h>

#endif //NEURALNETWORKUTILITY_H


class NeuralNetworkUtility
{
public:
    static std::vector<double> GetPredictedOutput(std::shared_ptr<NeuralDataObject> dataObject);
};