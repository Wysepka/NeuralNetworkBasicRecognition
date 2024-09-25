#pragma once
#include <vector>

class LayerBuffer
{
private:
	std::vector<double> valuesOriginal;
	std::vector<double> valuesGradient;
	std::vector<double> valuesActivation;
	std::vector<double> forwardWeights;

	LayerBuffer()
	{

	}
};