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

struct NeuralNetworkLogConfig
{
private:
	bool bUseLogPerIteration;
	bool bUseLogPerBatch;
	bool bUseLogPerEpoch;
public:
	NeuralNetworkLogConfig(bool useLogPerIteration, bool useLogPerBatch, bool useLogPerEpoch) :
	bUseLogPerIteration(useLogPerIteration), bUseLogPerBatch(useLogPerBatch) , bUseLogPerEpoch(useLogPerEpoch)
	{}

	NeuralNetworkLogConfig() : bUseLogPerIteration(false) , bUseLogPerBatch(false) , bUseLogPerEpoch(false)
	{}

	bool GetUseLogPerIteration() {return bUseLogPerIteration;}
	bool GetUseLogPerBatch() {return bUseLogPerBatch;}
	bool GetUseLogPerEpoch() {return bUseLogPerEpoch;}
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

	NeuralNetworkLogConfig logConfig;

public:
	static float LearningRateInitial;
	static float DecayRate;

	NeuralNetworkConfig(uint16_t inputLayer, std::vector<NeuralNetworkHiddenConfig> hiddenLayer, uint16_t outputLayer
		, std::shared_ptr<IActivation> outputActivation, std::shared_ptr<ICost> outputCost
		, uint32_t batchSize, uint32_t epochCount , bool useParallelBatchComp , uint16_t maxParallelBatchComputations
		, NeuralNetworkLogConfig loggingConfig)
		: inputLayer(inputLayer) , hiddenLayer(hiddenLayer) , outputLayer(outputLayer) , outputActivation(outputActivation)
		, outputCost(outputCost) , batchSize(batchSize) , epochCount(epochCount) , bUseParallelBatchComputation(useParallelBatchComp)
		, maxParallelBatchComputations(maxParallelBatchComputations) , logConfig(loggingConfig)
	{
	}

	uint16_t GetInputLayer() const;
	std::vector<NeuralNetworkHiddenConfig> GetHiddenLayer();
	uint16_t GetOutputLayer() const;
	std::shared_ptr<IActivation> GetOutputActivation();
	std::shared_ptr<ICost> GetOutputCost();

	uint32_t GetBatchSize() const;
	uint32_t GetEpochCount() const;
	bool GetUseParallelBatchComputation() const;
	uint16_t GetMaxParallelBatchComputation() const;

	NeuralNetworkLogConfig GetNeuralNetworkLogConfig() const;
};

