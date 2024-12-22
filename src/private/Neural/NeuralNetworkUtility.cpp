//
// Created by Marcin on 11/15/2024.
//

#include "Neural/NeuralNetworkUtility.h"
#include <Config/NeuralNetworkConfig.h>

std::vector<double> NeuralNetworkUtility::GetPredictedOutput(std::shared_ptr<NeuralDataObject> dataObject)
{
    std::vector<double> predictedOutput;
    for(size_t i = 0; i<10; i++)
    {
        if(dataObject->GetLabel() == i)
        {
            predictedOutput.push_back(1);
        }
        else
        {
            predictedOutput.push_back(0);
        }
    }
    return predictedOutput;
}

void NeuralNetworkUtility::GetHighestPropabilityPrediction(std::shared_ptr<LayerBuffer> outputLayerBuffer,
    int &predictedNum, float &chance) {

    double highestChance = -std::numeric_limits<double>::infinity();
    int highestChanceIndex = 0;

    for (size_t i = 0; i < outputLayerBuffer->valuesActivation.size(); i++)
    {
        if(outputLayerBuffer->valuesActivation[i] > highestChance)
        {
            highestChance = outputLayerBuffer->valuesActivation[i];
            highestChanceIndex = i;
        }
    }

    predictedNum = highestChanceIndex;
    chance = highestChance;
}

std::vector<std::shared_ptr<NeuralDataBatch>> NeuralNetworkUtility::SplitEpochToBatchVector(std::shared_ptr<NeuralDataFile> dataFile,
    uint32_t batchSize)
{
    int batchesCount = dataFile->GetNeuralDataObjects().size() / batchSize;

    std::vector<std::shared_ptr<NeuralDataBatch>> batches;
    batches.reserve(batchesCount);

    int batchIterator = 0;
    auto dataObjects = dataFile->GetNeuralDataObjects();

    std::vector<std::shared_ptr<NeuralDataObject>> objectCache;
    for(uint32_t i = 0; i < dataObjects.size(); i++)
    {
        objectCache.push_back(dataObjects[i]);
        batchIterator++;

        if(batchIterator >= batchSize)
        {
            auto neuralBatch = std::make_shared<NeuralDataBatch>(objectCache, i + 1 - objectCache.size());
            batches.push_back(neuralBatch);
            batchIterator = 0;
            objectCache.clear();
        }
    }

    return batches;
}

double NeuralNetworkUtility::Lerp(double a, double b, double t) {
    return a + t * (b - a);
}

double NeuralNetworkUtility::GetLearningRate(bool parallel, long long iterationID,long long batchID, long long epochID , long long dataTotalSize)
{
    if(parallel)
    {
        // Example: Decay the learning rate exponentially
        double epochFactor = std::exp(NeuralNetworkConfig::DecayRate * epochID);
        double batchFactor = 1.0 / (1.0 + NeuralNetworkConfig::DecayRate * batchID);
        double iterationFactor = 1.0 / (1.0 + NeuralNetworkConfig::DecayRate * iterationID);

        double finalResult = NeuralNetworkConfig::LearningRateInitial * epochFactor * batchFactor * iterationFactor;

        return finalResult;
    }
    else
    {
        return iterationID / dataTotalSize;
    }
}

double NeuralNetworkUtility::CalculateLearningRate_Basic(int currentEpochLoopIndex, int maxEpochCount, double initialLearningRate, double finalLearningRate) {
    if (currentEpochLoopIndex < 0 || currentEpochLoopIndex > maxEpochCount) {
        throw std::invalid_argument("currentEpochLoopIndex must be between 0 and maxEpochCount inclusive.");
    }

    // Linear decay formula
    double learningRate = initialLearningRate -
        ((initialLearningRate - finalLearningRate) * (static_cast<double>(currentEpochLoopIndex) / maxEpochCount));

    return learningRate;
}
