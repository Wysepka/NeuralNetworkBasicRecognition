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
