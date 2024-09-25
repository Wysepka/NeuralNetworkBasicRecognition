#pragma once
#include <vector>

struct NeuralNetworkConfig
{
private:
	uint16_t inputLayer;
	std::vector<NeuralNetworkHiddenConfig> hiddenLayer;
	uint16_t outputLayer;
public:
	NeuralNetworkConfig(uint16_t inputLayer, std::vector<NeuralNetworkHiddenConfig> hiddenLayer, uint16_t outputLayer)
		: inputLayer(inputLayer) , hiddenLayer(hiddenLayer) , outputLayer(outputLayer)
	{

	}
public:
	uint16_t GetInputLayer() { return inputLayer; }
	std::vector<NeuralNetworkHiddenConfig> GetHiddenLayer() { return hiddenLayer; }
	uint16_t GetOutputLayer() { return outputLayer; }
};

struct NeuralNetworkHiddenConfig
{
private:
	uint16_t nodesCount;
public:
	NeuralNetworkHiddenConfig(uint16_t nodesCount) : nodesCount(nodesCount)
	{

	}

	uint16_t GetNodesCount() { return nodesCount; }
};