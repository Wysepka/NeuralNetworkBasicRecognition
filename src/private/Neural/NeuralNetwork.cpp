
#include "Neural/NeuralNetwork.h"

#include "Event/MessageBus.h"

void NeuralNetwork::SetConfig(NeuralNetworkConfig config)
{
	auto hiddenLayerConfig = config.GetHiddenLayer();
	if (hiddenLayerConfig.size() == 0)
	{
		throw std::runtime_error("HiddenLayer size is 0 !");
		return;
	}

	inputLayer = std::make_shared<Layer>(config.GetInputLayer() , hiddenLayerConfig[0].GetNodesCount() , LayerType::Input);

	for (size_t i = 0; i < hiddenLayerConfig.size(); i++)
	{
		auto nextLayerNodesCount = i == hiddenLayerConfig.size() - 1 ? config.GetOutputLayer() : hiddenLayerConfig[i + 1].GetNodesCount();
		hiddenLayers.push_back(std::make_shared<Layer>(hiddenLayerConfig[i].GetNodesCount(), nextLayerNodesCount, LayerType::Hidden));
	}

	outputLayer = std::make_shared<Layer>(config.GetOutputLayer() , 0 , LayerType::Output);

	layersSize = 2;
	layersSize += hiddenLayerConfig.size();
	initialized = true;

	layersCombined.push_back(inputLayer);
	layersCombined.insert(layersCombined.end(), hiddenLayers.begin(), hiddenLayers.end());
	layersCombined.push_back(outputLayer);

	NeuralNetworkInitialized message(layersCombined);
	MessageBus::Publish(std::make_shared<NeuralNetworkInitialized>(message));
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
		std::vector<std::shared_ptr<LayerBuffer>> buffer = GetLayerBufferVector();

		auto data = *(datas[i]);
		FeedForward((datas[i])->GetFlatObjectPixelsArray_Normalized());
	}
}

std::vector<std::shared_ptr<LayerBuffer>> NeuralNetwork::GetLayerBufferVector()
{
	std::vector<std::shared_ptr<LayerBuffer>> buffer;
		
	buffer.push_back(std::make_shared<LayerBuffer>(inputLayer->ValuesCount() , hiddenLayers[0]->ValuesCount() , LayerType::Input));

	for(size_t i = 0; i < hiddenLayers.size(); i++)
	{
		int nodesOut = i == hiddenLayers.size() - 1 ? outputLayer->ValuesCount() : hiddenLayers[i + 1]->ValuesCount();
		buffer.push_back(std::make_shared<LayerBuffer>(hiddenLayers[i]->ValuesCount(), nodesOut , LayerType::Hidden));
	}

	buffer.push_back(std::make_shared<LayerBuffer>(outputLayer->ValuesCount() , 0 , LayerType::Output));

	//return buffer;
	return buffer;
}

void NeuralNetwork::FeedForward(std::vector<double> inputs)
{
	auto inputsToNextLayer = inputs;
	auto layerBuffer = GetLayerBufferVector();
	for (size_t i = 0; i < layersCombined.size(); i++)
	{
		inputsToNextLayer = layersCombined[i]->CalculateValues(inputsToNextLayer , layerBuffer[i]);
	}

	outputLayer->CalculateValues(inputsToNextLayer , layerBuffer[layerBuffer.size() - 1]);


	//outputLayer->CalculateValues(inputsToNextLayer , layerBuffer[0]);
}
