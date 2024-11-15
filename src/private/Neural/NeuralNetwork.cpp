
#include "Neural/NeuralNetwork.h"

#include "Event/MessageBus.h"
#include "Neural/NeuralNetworkUtility.h"

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

	for (size_t i = 0; i < layersCombined.size(); i++)
	{
		layersCombined[i]->SetActivationAndCost(config.GetOutputActivation() , config.GetOutputCost());
	}

	MessageBus::Publish<NeuralNetworkInitialized>(std::make_shared<NeuralNetworkInitialized>(layersCombined));
}

void NeuralNetwork::RunNetwork(std::shared_ptr<NeuralDataFile> dataFile)
{
	if (!initialized) 
	{
		return;
	}
	auto datas = dataFile->GetNeuralDataObjects();

	for (size_t i = 0; i < datas.size(); i++)
	{
		std::vector<std::shared_ptr<LayerBuffer>> buffer = GetLayerBufferVector();

		auto data = *(datas[i]);
		FeedForward((datas[i])->GetFlatObjectPixelsArray_Normalized() , buffer);
		Backpropagate(datas[i], buffer);
		UpdateNetwork();

		//NeuralNetworkResult result()

		//MessageBus::Publish<NeuralNetworkIterationMessage>()
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

void NeuralNetwork::FeedForward(std::vector<double> inputs , std::vector<std::shared_ptr<LayerBuffer>> bufferVector)
{
	auto inputsToNextLayer = inputs;
	for (size_t i = 0; i < layersCombined.size(); i++)
	{
		inputsToNextLayer = layersCombined[i]->CalculateValues(inputsToNextLayer , bufferVector[i]);
	}

	//Commented since we are calculating in loop above
	//outputLayer->CalculateValues(inputsToNextLayer , layerBuffer[layerBuffer.size() - 1]);


	//outputLayer->CalculateValues(inputsToNextLayer , layerBuffer[0]);
}

void NeuralNetwork::Backpropagate(std::shared_ptr<NeuralDataObject> dataObject, std::vector<std::shared_ptr<LayerBuffer>> bufferVector)
{

	std::vector<double> expectations = NeuralNetworkUtility::GetPredictedOutput(dataObject);
	outputLayer->CalculateOutputLayerGradient(expectations , bufferVector[bufferVector.size() - 1]);
	outputLayer->UpdateGradients(bufferVector[bufferVector.size() - 1]);

	for (size_t i = layersSize - 1; i >= 0; i--)
	{
		layersCombined[i]->CalculateHiddenLayerGradient(bufferVector[i] , layersCombined[i+1] , bufferVector[i+1]);
		layersCombined[i]->UpdateGradients(bufferVector[layersCombined.size() - 1]);
	}
	//outputLayer->CalculateOutputLayerGradient()
}

void NeuralNetwork::UpdateNetwork()
{

}
