#pragma once
#include <iostream>
#include "Data/NeuralDataFile.h"

class Logger 
{
public:
	void LogDataFile(const std::shared_ptr<const NeuralDataFile> dataFile);
};