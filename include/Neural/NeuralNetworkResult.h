//
// Created by Marcin on 11/15/2024.
//
#pragma once
#include <memory>
#include <vector>

class NeuralNetworkGroupResult
{
    public:
    NeuralNetworkGroupResult() : correctlyPredicted(0) , totalPredicted(0) , correctPercentage(0) , batchID(0) , epochID(0) {}

    long long correctlyPredicted;
    long long totalPredicted;

    std::vector<long long> predictions;
    std::vector<long long> totalPredictions;

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

    std::vector<std::shared_ptr<NeuralNetworkGroupResult>> batchResults;
    std::vector<std::shared_ptr<NeuralNetworkGroupResult>> epochResults;

    NeuralNetworkResult(long long int testedCount, long long int positiveResultCount, long long int totalCount) : testedCount(testedCount) , positiveResultCount(positiveResultCount) , totalCount(totalCount) {};
    NeuralNetworkResult() { testedCount = positiveResultCount = totalCount = 0; };
};