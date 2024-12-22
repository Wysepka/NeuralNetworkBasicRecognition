//
// Created by Marcin on 11/15/2024.
//
#pragma once
#include <memory>
#include <vector>

class NeuralNetworkGroupResult
{
    public:
    NeuralNetworkGroupResult() : correctlyPredicted(0) , totalPredicted(0) , correctPercentage(0) , batchID(0) , epochID(0)
    {
        for (size_t i = 0; i < 10; i++)
        {
            predictions.push_back(0);
            actualNumbers.push_back(0);
        }
    }

    long long correctlyPredicted;
    long long totalPredicted;

    std::vector<long long> predictions;
    std::vector<long long> actualNumbers;

    double correctPercentage;
    long long batchID;
    long long epochID;

};

class NeuralNetworkResult
{
public:
    long long int testedCount;
    long long int positiveResultCount;
    long long int totalCount;

    std::vector<long long> correctResults;
    std::vector<long long> allResults;

    std::vector<std::shared_ptr<NeuralNetworkGroupResult>> batchResults;
    std::vector<std::shared_ptr<NeuralNetworkGroupResult>> epochResults;

    long long epochCount;

    NeuralNetworkResult(long long int testedCount, long long int positiveResultCount, long long int totalCount)
    : testedCount(testedCount) , positiveResultCount(positiveResultCount) , totalCount(totalCount)
    {
        InitializeVectors();
    };
    NeuralNetworkResult() { testedCount = positiveResultCount = totalCount = 0; InitializeVectors();};

    void InitializeVectors()
    {
        for (size_t i = 0; i < 10; i++)
        {
            correctResults.push_back(0);
            allResults.push_back(0);
        }
    }
};