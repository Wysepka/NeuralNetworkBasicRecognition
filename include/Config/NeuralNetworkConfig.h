#pragma once
#include <vector>
#include <Neural/IActivation.h>
#include <Neural/ICost.h>

struct NeuralNetworkHiddenConfig
{
private:
	uint16_t nodesCount;
	std::shared_ptr<IActivation> activation;
	std::shared_ptr<ICost> cost;
public:
	NeuralNetworkHiddenConfig(uint16_t nodesCount) : nodesCount(nodesCount)
	{

	}

	uint16_t GetNodesCount() { return nodesCount; }
};

struct NeuralNetworkConfig
{
private:
	uint16_t inputLayer;
	std::vector<NeuralNetworkHiddenConfig> hiddenLayer;
	uint16_t outputLayer;

	std::shared_ptr<IActivation> outputActivation;
	std::shared_ptr<ICost> outputCost;

	uint32_t batchSize;
	uint32_t epochCount;

	uint16_t maxParallelBatchComputations;
	bool bUseParallelBatchComputation;

public:
	NeuralNetworkConfig(uint16_t inputLayer, std::vector<NeuralNetworkHiddenConfig> hiddenLayer, uint16_t outputLayer
		, std::shared_ptr<IActivation> outputActivation, std::shared_ptr<ICost> outputCost
		, uint32_t batchSize, uint32_t epochCount , bool useParallelBatchComp , uint16_t maxParallelBatchComputations)
		: inputLayer(inputLayer) , hiddenLayer(hiddenLayer) , outputLayer(outputLayer) , outputActivation(outputActivation)
		, outputCost(outputCost) , batchSize(batchSize) , epochCount(epochCount) , bUseParallelBatchComputation(useParallelBatchComp)
		, maxParallelBatchComputations(maxParallelBatchComputations)
	{

	}
public:
	uint16_t GetInputLayer() { return inputLayer; }
	std::vector<NeuralNetworkHiddenConfig> GetHiddenLayer() { return hiddenLayer; }
	uint16_t GetOutputLayer() { return outputLayer; }
	std::shared_ptr<IActivation> GetOutputActivation() { return outputActivation; }
	std::shared_ptr<ICost> GetOutputCost() { return outputCost; }

	uint32_t GetBatchSize() {return batchSize;}
	uint32_t GetEpochCount() {return epochCount;}
	bool GetUseParallelBatchComputation() {return bUseParallelBatchComputation;}
	uint16_t GetMaxParallelBatchComputation() {return maxParallelBatchComputations;}
};