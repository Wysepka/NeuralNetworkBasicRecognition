
#include "Neural/NeuralNetwork.h"

#include <iostream>
#include <ostream>

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

	/*
	inputLayer = std::make_shared<Layer>(config.GetInputLayer() , hiddenLayerConfig[0].GetNodesCount() , LayerType::Input);

	for (size_t i = 0; i < hiddenLayerConfig.size(); i++)
	{
		auto nextLayerNodesCount = i == hiddenLayerConfig.size() - 1 ? config.GetOutputLayer() : hiddenLayerConfig[i + 1].GetNodesCount();
		hiddenLayers.push_back(std::make_shared<Layer>(hiddenLayerConfig[i].GetNodesCount(), nextLayerNodesCount, LayerType::Hidden));
	}

	outputLayer = std::make_shared<Layer>(config.GetOutputLayer() , 0 , LayerType::Output);

	*/

	for (size_t i = 0; i < hiddenLayerConfig.size(); i++)
	{
		int nodesIn = i == 0 ? config.GetInputLayer() : hiddenLayerConfig[i - 1].GetNodesCount();
		int nodesOut = hiddenLayerConfig[i].GetNodesCount();

		hiddenLayers.push_back(std::make_shared<Layer>(nodesIn , nodesOut , LayerType::Hidden));
		//auto nextLayerNodesCount = i == hiddenLayerConfig.size() - 1 ? config.GetOutputLayer() : hiddenLayerConfig[i + 1].GetNodesCount();
		//hiddenLayers.push_back(std::make_shared<Layer>(hiddenLayerConfig[i].GetNodesCount(), nextLayerNodesCount, LayerType::Hidden));
	}

	outputLayer = std::make_shared<Layer>(hiddenLayerConfig[hiddenLayerConfig.size() - 1].GetNodesCount() , 10 , LayerType::Output);

	//This is currently not taking InputLayer as layer in consideration
	//layersSize = 2;
	layersSize = 1;
	layersSize += hiddenLayerConfig.size();
	initialized = true;

	//layersCombined.push_back(inputLayer);
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

	int correctlyPredicted = 0;

	for (size_t i = 0; i < datas.size(); i++)
	{
		std::vector<std::shared_ptr<LayerBuffer>> buffer = GetLayerBufferVector();

		auto data = *(datas[i]);
		FeedForward((datas[i])->GetFlatObjectPixelsArray_Normalized() , buffer);
		Backpropagate(datas[i], buffer);
		UpdateNetwork();

		int predictedNum = 0;
		float predictionChance = 0.f;

		NeuralNetworkUtility::GetHighestPropabilityPrediction(buffer[buffer.size() - 1] , predictedNum , predictionChance);

		//std::cout << "Num Prediction is "  <<predictedNum << " | Label is "<<  std::endl;
		//std::cout << "===============================================" << std::endl;

		bool correctPrediction = predictedNum == datas[i]->GetLabel();
		if (correctPrediction)
		{
			correctlyPredicted++;
		}
		std::cout << "Correct Prediction Percentage: " << correctlyPredicted / (i+1) << std::endl;

		//NeuralNetworkResult result()

		//MessageBus::Publish<NeuralNetworkIterationMessage>()
	}
}

std::vector<std::shared_ptr<LayerBuffer>> NeuralNetwork::GetLayerBufferVector()
{
	std::vector<std::shared_ptr<LayerBuffer>> buffer;
		
	//buffer.push_back(std::make_shared<LayerBuffer>(inputLayer->ValuesCount() , hiddenLayers[0]->ValuesCount() , LayerType::Input));

	for(size_t i = 0; i < hiddenLayers.size(); i++)
	{
		int nodesIn = hiddenLayers[i]->NodesIn();
		int nodesOut = hiddenLayers[i]->NodesOut();
		//int nodesOut = i == hiddenLayers.size() - 1 ? outputLayer->ValuesCount() : hiddenLayers[i + 1]->ValuesCount();
		buffer.push_back(std::make_shared<LayerBuffer>(nodesIn, nodesOut , LayerType::Hidden));
	}

	buffer.push_back(std::make_shared<LayerBuffer>(outputLayer->NodesIn() , outputLayer->NodesOut() , LayerType::Output));

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

	for (int i = layersSize - 2; i >= 0; i--)
	{
		layersCombined[i]->CalculateHiddenLayerGradient(bufferVector[i] , layersCombined[i+1] , bufferVector[i+1]);
		layersCombined[i]->UpdateGradients(bufferVector[i]);
	}
	//outputLayer->CalculateOutputLayerGradient()
}

void NeuralNetwork::UpdateNetwork()
{
	for (size_t i = 0; i < layersCombined.size(); i++)
	{
		//Implement NeuralPreferences struct as passing for params
		layersCombined[i]->ApplyGradients(0.01f, 0.1f , 0.9f);
	}
}
