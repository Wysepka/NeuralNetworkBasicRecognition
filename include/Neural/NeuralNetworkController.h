//
// Created by Marcin on 11/15/2024.
//

#ifndef NEURALNETWORKCONTROLLER_H
#define NEURALNETWORKCONTROLLER_H

#endif //NEURALNETWORKCONTROLLER_H

#include <memory>
#include "NeuralNetwork.h"
#include "Activation.h"
#include "Cost.h"

class NeuralNetworkController
{
private:
    std::shared_ptr<bool> isCurrentlyRunning;
    std::shared_ptr<NeuralNetwork> network;
    std::shared_ptr<NeuralDataFile> neuralDataFile;
public:
    void Initialize(std::shared_ptr<NeuralDataFile> neuralDataFile);
    void Run(bool& shouldBeRunning);
    void Dispose();
};