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
public:
	NeuralNetworkConfig(uint16_t inputLayer, std::vector<NeuralNetworkHiddenConfig> hiddenLayer, uint16_t outputLayer , std::shared_ptr<IActivation> outputActivation, std::shared_ptr<ICost> outputCost)
		: inputLayer(inputLayer) , hiddenLayer(hiddenLayer) , outputLayer(outputLayer) , outputActivation(outputActivation) , outputCost(outputCost)
	{

	}
public:
	uint16_t GetInputLayer() { return inputLayer; }
	std::vector<NeuralNetworkHiddenConfig> GetHiddenLayer() { return hiddenLayer; }
	uint16_t GetOutputLayer() { return outputLayer; }
	std::shared_ptr<IActivation> GetOutputActivation() { return outputActivation; }
	std::shared_ptr<ICost> GetOutputCost() { return outputCost; }
};