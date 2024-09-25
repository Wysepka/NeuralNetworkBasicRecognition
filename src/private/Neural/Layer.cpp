#include "Neural/Layer.h"

void Layer::InitializeRandomWeights(std::mt19937& rng)
{
	for (size_t i = 0; i < weights.size(); i++)
	{
		weights[i] = RandomInNormalDistribution(rng, 0.0, 1.0) / std::sqrt(values.size());
	}
}

double Layer::RandomInNormalDistribution(std::mt19937& rng, double mean, double standardDeviation)
{
	std::uniform_real_distribution<> dist(0.0, 1.0);

	double x1 = 1.0 - dist(rng);
	double x2 = 1.0 - dist(rng);

	double y1 = std::sqrt(-2.0 * std::log(x1)) * std::cos(2.0 * M_PI * x2);
	return y1 * standardDeviation + mean;
}

std::vector<double> Layer::CalculateValues(std::vector<double> inputs)
{
	if (inputs.size() != values.size()) 
	{
		throw std::runtime_error("Inputs Size is different than Values size in input Layer !");
		return;
	}
	for (size_t i = 0; i < inputs.size(); i++)
	{
		values
	}
}
