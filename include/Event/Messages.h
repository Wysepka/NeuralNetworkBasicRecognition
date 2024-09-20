#pragma once
#include "memory"
#include "Data/NeuralDataFile.h"

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