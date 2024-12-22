//
// Created by Marcin on 11/15/2024.
//

#ifndef NEURALNETWORKUTILITY_H
#define NEURALNETWORKUTILITY_H
#include <vector>
#include <cmath>
#include <stdexcept>
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
    static double Lerp(double a, double b, double t);
    static double GetLearningRate(bool parallel, long long iterationID,long long batchID, long long epochID , long long dataTotalSize);
    static double CalculateLearningRate_Basic(int currentEpochLoopIndex, int maxEpochCount, double initialLearningRate = 0.1, double finalLearningRate = 0.001);

};