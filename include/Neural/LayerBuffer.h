#pragma once
#include <vector>

#include "Core/EnumsContainer.h"

class LayerBuffer
{
public:
	std::vector<double> valuesOriginal;
	std::vector<double> valuesCalculated;
	std::vector<double> valuesGradient;
	std::vector<double> valuesActivation;

	int nodesIn; 
	int nodesOut;

	LayerType layerType;

	LayerBuffer(int nodesIn, int nodesOut , LayerType layerType) : nodesIn(nodesIn) , nodesOut(nodesOut) , layerType(layerType)
	{
		valuesOriginal = std::vector<double>(nodesOut);
		valuesCalculated = std::vector<double>(nodesOut);
		valuesGradient = std::vector<double>(nodesOut);
		valuesActivation = std::vector<double>(nodesOut);
	}
};
