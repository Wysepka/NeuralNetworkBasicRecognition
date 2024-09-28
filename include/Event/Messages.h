#pragma once
#include "memory"
#include "Data/NeuralDataFile.h"
#include "Neural/Layer.h"

class Messages
{

};

struct Message
{
	virtual ~Message() = default;
};

struct EndLoadingFile : Message
{
public:
	std::shared_ptr<NeuralDataFile> neuralDataFile;

	EndLoadingFile(std::shared_ptr<NeuralDataFile> neuralDataFile) : neuralDataFile(neuralDataFile)
	{

	};
};

struct NeuralNetworkInitialized : Message {

public:
	std::vector<std::shared_ptr<Layer>> layerVector;

	NeuralNetworkInitialized(std::vector<std::shared_ptr<Layer>> layerVector) : layerVector(layerVector){};
};