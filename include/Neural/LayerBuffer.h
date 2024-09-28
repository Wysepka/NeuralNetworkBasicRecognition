#pragma once
#include <vector>

#include "Core/EnumsContainer.h"

class LayerBuffer
{
public:
	std::vector<double> valuesOriginal;
	std::vector<double> valuesGradient;
	std::vector<double> valuesActivation;
	std::vector<double> forwardWeights;

	int nodesIn; 
	int nodesOut;

	LayerType layerType;

	LayerBuffer(int nodesIn, int nodesOut , LayerType layerType) : nodesIn(nodesIn) , nodesOut(nodesOut) , layerType(layerType)
	{

	}
};
