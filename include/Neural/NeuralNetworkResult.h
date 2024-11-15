//
// Created by Marcin on 11/15/2024.
//

#ifndef NEURALNETWORKRESULT_H
#define NEURALNETWORKRESULT_H

#endif //NEURALNETWORKRESULT_H

class NeuralNetworkResult
{
public:
    long long int testedCount;
    long long int positiveResultCount;
    long long int totalCount;

    NeuralNetworkResult(long long int testedCount, long long int positiveResultCount, long long int totalCount) : testedCount(testedCount) , positiveResultCount(positiveResultCount) , totalCount(totalCount) {};
};