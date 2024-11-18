#pragma once
#include <cstdint>
#include <memory>
#include <stdexcept>
#include "Config/NeuralNetworkConfig.h"
#include "Neural/Layer.h"
#include "Neural/LayerBuffer.h"
#include "Data/NeuralDataFile.h"
#include "Event/MessageBus.h"

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

	std::vector<std::shared_ptr<LayerBuffer>> GetLayerBufferVector();

	void FeedForward(std::vector<double> inputs, std::vector<std::shared_ptr<LayerBuffer>> bufferVector);
	void Backpropagate(std::shared_ptr<NeuralDataObject> dataObject, std::vector<std::shared_ptr<LayerBuffer>> bufferVector);
	void UpdateNetwork(double learningRate);
public:
	void SetConfig(NeuralNetworkConfig config);
	void RunNetwork(std::shared_ptr<NeuralDataFile> dataFile);
};

