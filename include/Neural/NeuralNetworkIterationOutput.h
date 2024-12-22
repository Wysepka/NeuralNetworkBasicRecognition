//
// Created by Marcin on 12/22/2024.
//

#ifndef NEURALNETWORKITERATIONOUTPUT_H
#define NEURALNETWORKITERATIONOUTPUT_H

struct NeuralNetworkIterationOutput
{
public:
    NeuralNetworkIterationOutput();
    int predictedNumber;
    int actualNumber;
    double propabilityChance;
    bool predictedSuccesfully;
};

#endif //NEURALNETWORKITERATIONOUTPUT_H
