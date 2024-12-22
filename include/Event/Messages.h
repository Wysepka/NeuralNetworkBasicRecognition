#pragma once
#include "memory"
#include "Data/NeuralDataFile.h"
#include "Neural/Layer.h"
#include "Neural/NeuralNetworkResult.h"

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

struct NeuralNetworkIterationMessage : Message
{
public:
	std::shared_ptr<NeuralNetworkResult> neuralNetworkResult;
	NeuralNetworkIterationMessage(std::shared_ptr<NeuralNetworkResult> neuralNetworkResult) : neuralNetworkResult(neuralNetworkResult){};
};

struct NeuralNetworkBatchMessage : Message
{
	public:
	std::shared_ptr<NeuralNetworkResult> neuralNetworkResult;
	NeuralNetworkBatchMessage(std::shared_ptr<NeuralNetworkResult> neuralNetworkResult) : neuralNetworkResult(neuralNetworkResult){};
};