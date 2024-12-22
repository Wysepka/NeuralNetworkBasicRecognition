//
// Created by Marcin on 12/21/2024.
//
#include "Config/NeuralNetworkConfig.h"

float NeuralNetworkConfig::LearningRateInitial = 0.1f;
float NeuralNetworkConfig::DecayRate = 0.01f;

uint16_t NeuralNetworkConfig::GetInputLayer() const { return inputLayer; }
std::vector<NeuralNetworkHiddenConfig> NeuralNetworkConfig::GetHiddenLayer() { return hiddenLayer; }
uint16_t NeuralNetworkConfig::GetOutputLayer() const { return outputLayer; }
std::shared_ptr<IActivation> NeuralNetworkConfig::GetOutputActivation() { return outputActivation; }
std::shared_ptr<ICost> NeuralNetworkConfig::GetOutputCost() { return outputCost; }

uint32_t NeuralNetworkConfig::GetBatchSize() const {return batchSize;}
uint32_t NeuralNetworkConfig::GetEpochCount() const {return epochCount;}
bool NeuralNetworkConfig::GetUseParallelBatchComputation() const {return bUseParallelBatchComputation;}
uint16_t NeuralNetworkConfig::GetMaxParallelBatchComputation() const {return maxParallelBatchComputations;}

NeuralNetworkLogConfig NeuralNetworkConfig::GetNeuralNetworkLogConfig() const {return logConfig;}