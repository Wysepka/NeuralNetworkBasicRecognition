
#include "Neural/NeuralNetwork.h"
#include "Event/MessageBus.h"
#include "Neural/NeuralNetworkUtility.h"

void NeuralNetwork::SetConfig(NeuralNetworkConfig config)
{
	this->logConfig = config.GetNeuralNetworkLogConfig();
	auto hiddenLayerConfig = config.GetHiddenLayer();
	if (hiddenLayerConfig.size() == 0)
	{
		throw std::runtime_error("HiddenLayer size is 0 !");
		return;
	}

	for (size_t i = 0; i < hiddenLayerConfig.size(); i++)
	{
		int nodesIn = i == 0 ? config.GetInputLayer() : hiddenLayerConfig[i - 1].GetNodesCount();
		int nodesOut = hiddenLayerConfig[i].GetNodesCount();

		hiddenLayers.push_back(std::make_shared<Layer>(nodesIn , nodesOut , LayerType::Hidden));
	}

	outputLayer = std::make_shared<Layer>(hiddenLayerConfig[hiddenLayerConfig.size() - 1].GetNodesCount() , 10 , LayerType::Output);

	//This is currently not taking InputLayer as layer in consideration
	//layersSize = 2;
	layersSize = 1;
	layersSize += hiddenLayerConfig.size();

	batchSize = config.GetBatchSize();
	epochCount = config.GetEpochCount();
	bRunParallel = config.GetUseParallelBatchComputation();
	batchMaxComputeCount = config.GetMaxParallelBatchComputation();

	//layersCombined.push_back(inputLayer);
	layersCombined.insert(layersCombined.end(), hiddenLayers.begin(), hiddenLayers.end());
	layersCombined.push_back(outputLayer);

	for (size_t i = 0; i < layersCombined.size(); i++)
	{
		layersCombined[i]->SetActivationAndCost(config.GetOutputActivation() , config.GetOutputCost());
	}

	MessageBus::Publish<NeuralNetworkInitialized>(std::make_shared<NeuralNetworkInitialized>(layersCombined));

	initialized = true;
}

void NeuralNetwork::RunNetwork(std::shared_ptr<NeuralDataFile> dataFile)
{
	if (!initialized) 
	{
		return;
	}
	auto datas = dataFile->GetNeuralDataObjects();

	int correctlyPredicted = 0;
	std::shared_ptr<NeuralNetworkResult> networkResult = std::make_shared<NeuralNetworkResult>();

	networkResult->totalCount = datas.size() * epochCount;

	for (size_t i = 0; i < epochCount; i++) {
		std::vector<std::shared_ptr<NeuralDataBatch>> batches = NeuralNetworkUtility::SplitEpochToBatchVector(dataFile , batchSize);

		for (size_t j = 0; j < batches.size(); j++)
		{
			std::shared_ptr<NeuralNetworkGroupResult> batchResult = std::make_shared<NeuralNetworkGroupResult>();
			batchResult->batchID = j;
			networkResult->batchResults.push_back(batchResult);
			batches[i]->groupResultRef = batchResult;
		}

		auto epochResult = std::make_shared<NeuralNetworkGroupResult>();
		epochResult->epochID = i;
		networkResult->epochResults.push_back(epochResult);

		if(bRunParallel)
		{
			RunBatchesParallel(batches , networkResult);
		}
		else
		{
			IterateThroughAllDataObjects(datas, networkResult);
		}
	}
}

void NeuralNetwork::RunBatchesParallel(std::vector<std::shared_ptr<NeuralDataBatch>> batchesVector , std::shared_ptr<NeuralNetworkResult> networkResult)
{
	int batchesCompleted = 0;
	int batchesRunning = 0;

	while (batchesCompleted < batchesVector.size()) {
		std::vector<std::thread> batchesThreads;
		int batchesToRun = std::min<int>(batchMaxComputeCount, static_cast<int>(batchesVector.size() - batchesCompleted));

		for (int i = 0; i < batchesToRun; ++i) {
			// Start a new thread for each batch
			batchesThreads.emplace_back(&NeuralNetwork::RunBatchIteration, this, batchesVector[batchesCompleted + i], networkResult);
		}

		for (auto& thread : batchesThreads) {
			if (thread.joinable()) {
				thread.join(); // Wait for all threads to finish
			}
		}

		batchesCompleted += batchesToRun;
	}
}

void NeuralNetwork::RunBatchIteration(std::shared_ptr<NeuralDataBatch> batchData, std::shared_ptr<NeuralNetworkResult> networkResult)
{
	auto neuralDataObjects = batchData->GetNeuralDataObjects();
	for (size_t i = 0; i < neuralDataObjects.size(); i++)
	{
		RunSingleTrainingIterationThroughNetwork(neuralDataObjects[i] , batchData->GetIterationIDStart() , networkResult);
	}
}

void NeuralNetwork::IterateThroughAllDataObjects(std::vector<std::shared_ptr<NeuralDataObject>> datas , std::shared_ptr<NeuralNetworkResult> result) {
	for (size_t i = 0; i < datas.size(); i++)
	{
		RunSingleTrainingIterationThroughNetwork(datas[i], i , result);
	}
}

void NeuralNetwork::RunSingleTrainingIterationThroughNetwork(std::shared_ptr<NeuralDataObject> data, size_t iterationID , std::shared_ptr<NeuralNetworkResult> result)
{
	/*
	std::vector<std::shared_ptr<LayerBuffer>> buffer = GetLayerBufferVector();

	//auto data = *(datas[i]);
	FeedForward((data)->GetFlatObjectPixelsArray_Normalized() , buffer);
	Backpropagate(data, buffer);
	UpdateNetwork();

	int predictedNum = 0;
	float predictionChance = 0.f;

	NeuralNetworkUtility::GetHighestPropabilityPrediction(buffer[buffer.size() - 1] , predictedNum , predictionChance);

	//std::cout << "Num Prediction is "  <<predictedNum << " | Label is "<<  std::endl;
	//std::cout << "===============================================" << std::endl;

	bool correctPrediction = predictedNum == data->GetLabel();
	if (correctPrediction)
	{
		correctlyPredicted++;
	}
	std::cout << "Correct Prediction Percentage: " << correctlyPredicted / (iterationID+1) << std::endl;

	//NeuralNetworkResult result()

	//MessageBus::Publish<NeuralNetworkIterationMessage>()
	*/

	std::vector<std::shared_ptr<LayerBuffer>> buffer = GetLayerBufferVector();

	//auto data = *(datas);
	FeedForward((data)->GetFlatObjectPixelsArray_Normalized() , buffer);
	Backpropagate(data, buffer);

	double iterationDouble = static_cast<double>(iterationID);
	double dataSizeDouble = static_cast<double>(result->totalCount);
	double iterationPercentage = iterationDouble / dataSizeDouble;

	float learningRate = 0.01f;
	//float learningRate = NeuralNetworkUtility::Lerp(0.1f , 0.01f, iterationPercentage);
	if(bRunParallel)
	{
		int batchID = result->batchResults[result->batchResults.size() - 1]->batchID;
		int epochID = result->epochResults[result->epochResults.size() - 1]->epochID;
		learningRate = NeuralNetworkUtility::GetLearningRate(bRunParallel , iterationID , batchID , epochID , result->totalCount);
	}
	else
	{
		learningRate = NeuralNetworkUtility::GetLearningRate(bRunParallel , iterationID , 0 , 0 ,result->totalCount);
	}

	UpdateNetwork(learningRate);

	int predictedNum = 0;
	float predictionChance = 0.f;

	std::vector<double> propabilityChances = buffer[buffer.size() - 1]->valuesActivation;

	NeuralNetworkUtility::GetHighestPropabilityPrediction(buffer[buffer.size() - 1] , predictedNum , predictionChance);

	//std::cout << "Num Prediction is "  <<predictedNum << " | Label is "<<  std::endl;
	//std::cout << "===============================================" << std::endl;

	int label = data->GetLabel();
	result->testedCount++;

	bool correctPrediction = predictedNum == label;
	if (correctPrediction)
	{
		result->positiveResultCount++;
	}

	if(bRunParallel)
	{
		int batchResultID = result->batchResults.size() -1;
		result->batchResults[batchResultID]->totalPredicted++;
		result->batchResults[batchResultID]->predictions[predictedNum];
		result->batchResults[batchResultID]->totalPredictions[label];
	}

	if(logConfig.GetUseLogPerIteration())
	{
		float currentPredictionPercentage = static_cast<float>(result->positiveResultCount) / static_cast<float>(iterationID+1);
		std::cout << "Correct Prediction Percentage: " << currentPredictionPercentage << '%' << std::endl;

		for (size_t i = 0; i < propabilityChances.size(); i++)
		{
			std::cout << "For Label: " << i << "| Chance: " << propabilityChances[i] << std::endl;
		}
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

void NeuralNetwork::UpdateNetwork(double learningRate)
{
	for (size_t i = 0; i < layersCombined.size(); i++)
	{
		//Implement NeuralPreferences struct as passing for params
		layersCombined[i]->ApplyGradients(learningRate, 0.1f , 0.9f);
	}
}
