//
// Created by Marcin on 11/15/2024.
//

#ifndef NEURALNETWORKUTILITY_H
#define NEURALNETWORKUTILITY_H
#include <vector>
#include <Data/NeuralDataFile.h>

#include "LayerBuffer.h"
#include "Data/NeuralDataBatch.h"

#endif //NEURALNETWORKUTILITY_H


class NeuralNetworkUtility
{
public:
    static std::vector<double> GetPredictedOutput(std::shared_ptr<NeuralDataObject> dataObject);
    static void GetHighestPropabilityPrediction(std::shared_ptr<LayerBuffer> outputLayerBuffer , int& predictedNum, float& chance);

    static std::vector<std::shared_ptr<NeuralDataBatch>> SplitEpochToBatchVector(std::shared_ptr<NeuralDataFile> dataFile, uint32_t batchSize);
};