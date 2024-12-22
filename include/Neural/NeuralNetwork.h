#pragma once
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>
#include <iostream>
#include <ostream>

#include "NeuralNetworkIterationOutput.h"
#include "Config/NeuralNetworkConfig.h"
#include "Neural/Layer.h"
#include "Neural/LayerBuffer.h"
#include "Data/NeuralDataFile.h"
#include "Event/MessageBus.h"
#include "Data/NeuralDataBatch.h"
#include "NeuralNetworkResult.h"

class NeuralNetwork
{
private:
	std::shared_ptr<Layer> inputLayer;
	std::vector<std::shared_ptr<Layer>> hiddenLayers;
	std::shared_ptr<Layer> outputLayer;

	std::vector<std::shared_ptr<Layer>> layersCombined;

	NeuralNetworkLogConfig logConfig;
	//long long iterationID,long long batchID, long long epochID , long long dataTotalSize , float decayRate
	//Function for calculating LearningRate at current iteration
	std::function<float(long long,long long, long long, long long, float)> getLearningRateFnc;

	NeuralDataFile dataFile;

	bool initialized;
	int layersSize;

	uint32_t batchSize;
	uint32_t epochCount;
	bool bRunParallel;
	uint32_t batchMaxComputeCount;

	std::vector<std::shared_ptr<LayerBuffer>> GetLayerBufferVector();

	void FeedForward(std::vector<double> inputs, std::vector<std::shared_ptr<LayerBuffer>> bufferVector);
	void Backpropagate(std::shared_ptr<NeuralDataObject> dataObject, std::vector<std::shared_ptr<LayerBuffer>> bufferVector);
	void UpdateNetwork(double learningRate);
public:
	void SetConfig(NeuralNetworkConfig config);

	void RunBatchesParallel(std::vector<std::shared_ptr<NeuralDataBatch>> batches , std::shared_ptr<NeuralNetworkResult> networkResult);
	void RunBatchIteration(std::shared_ptr<NeuralDataBatch> batchData , std::shared_ptr<NeuralNetworkResult> networkResult);

	NeuralNetworkIterationOutput RunSingleTrainingIterationThroughNetwork(std::shared_ptr<NeuralDataObject> data, size_t iterationID , std::shared_ptr<NeuralNetworkResult> result);

	void IterateThroughAllDataObjects(std::vector<std::shared_ptr<NeuralDataObject>> datas , std::shared_ptr<NeuralNetworkResult> result);

	void RunNetwork(std::shared_ptr<NeuralDataFile> dataFile);
};

