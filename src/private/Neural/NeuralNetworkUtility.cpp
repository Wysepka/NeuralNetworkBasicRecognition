//
// Created by Marcin on 11/15/2024.
//

#include "Neural/NeuralNetworkUtility.h"

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

    float highestChance = 0;
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

double NeuralNetworkUtility::Lerp(double a, double b, double t) {
    return a + t * (b - a);
}
