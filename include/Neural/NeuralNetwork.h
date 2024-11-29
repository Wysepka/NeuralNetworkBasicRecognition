#pragma once
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>
#include <iostream>
#include <ostream>
#include "Config/NeuralNetworkConfig.h"
#include "Neural/Layer.h"
#include "Neural/LayerBuffer.h"
#include "Data/NeuralDataFile.h"
#include "Event/MessageBus.h"
#include "Data/NeuralDataBatch.h"

class NeuralNetwork
{
private:
	std::shared_ptr<Layer> inputLayer;
	std::vector<std::shared_ptr<Layer>> hiddenLayers;
	std::shared_ptr<Layer> outputLayer;

	std::vector<std::shared_ptr<Layer>> layersCombined;

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
	void UpdateNetwork();
public:
	void SetConfig(NeuralNetworkConfig config);

	void RunBatchesParallel(std::vector<std::shared_ptr<NeuralDataBatch>> batches);
	void RunBatchIteration(std::shared_ptr<NeuralDataBatch> batchData);

	void RunSingleTrainingIterationThroughNetwork(std::shared_ptr<NeuralDataObject> data,
	                                              int correctlyPredicted, size_t iterationID);

	void IterateThroughAllDataObjects(std::vector<std::shared_ptr<NeuralDataObject>> datas, int correctlyPredicted);

	void RunNetwork(std::shared_ptr<NeuralDataFile> dataFile);
};

