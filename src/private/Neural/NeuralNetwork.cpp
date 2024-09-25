#include "NeuralNetwork.h"
#include "NeuralNetwork.h"
#include "NeuralNetwork.h"
#include "Neural/NeuralNetwork.h"

void NeuralNetwork::SetConfig(NeuralNetworkConfig config)
{
	auto hiddenLayerConfig = config.GetHiddenLayer();
	if (hiddenLayerConfig.size() == 0)
	{
		throw std::runtime_error("HiddenLayer size is 0 !");
		return;
	}

	inputLayer = std::make_shared<Layer>(config.GetInputLayer());

	for (size_t i = 0; i < hiddenLayerConfig.size(); i++)
	{
		auto nextLayerNodesCount = i == hiddenLayerConfig.size() - 1 ? config.GetOutputLayer() : hiddenLayerConfig[i + 1].GetNodesCount();
		hiddenLayers.push_back(std::make_shared<Layer>(hiddenLayerConfig[i].GetNodesCount(), nextLayerNodesCount));
	}

	outputLayer = std::make_shared<Layer>(config.GetOutputLayer());

	layersSize = 2;
	layersSize += hiddenLayerConfig.size();
	initialized = true;
}

void NeuralNetwork::RunNetwork(NeuralDataFile dataFile)
{
	if (!initialized) 
	{
		return;
	}
	auto datas = dataFile.GetNeuralDataObjects();

	for (size_t i = 0; i < datas.size(); i++)
	{
		std::vector<LayerBuffer> buffer = GetLayerBuffer();

		auto data = *(datas[i]);
		FeedForward((datas[i])->GetFlatObjectPixelsArray_Normalized());
	}
}


std::vector<LayerBuffer> NeuralNetwork::GetLayerBuffer()
{
	std::vector<LayerBuffer> buffer;

	buffer.push_back()

	return buffer;
}

void NeuralNetwork::FeedForward(std::vector<double> inputs)
{

}
