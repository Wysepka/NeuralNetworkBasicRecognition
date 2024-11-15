#include "Neural/Layer.h"

void Layer::InitializeRandomWeights(std::mt19937& rng)
{
	for (size_t i = 0; i < weightsForward.size(); i++)
	{
		weightsForward[i] = RandomInNormalDistribution(rng, 0.0, 1.0) / std::sqrt(values.size());
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

double Layer::GetWeight(unsigned int nodeIn, unsigned int nodeOut)
{
	int index = nodesIn * nodeOut + nodeIn;
	return weightsForward[index];
}

std::vector<double> Layer::CalculateValues(std::vector<double> inputs, std::shared_ptr<LayerBuffer> layerBuffer)
{
	std::vector<double> outputs;
	if (inputs.size() != values.size()) 
	{
		throw std::runtime_error("Inputs Size is different than Values size in input Layer !");
		return outputs;
	}
	for (size_t i = 0; i < nodesOut; i++)
	{
		double valueBiased = biases[i];
		for (size_t j = 0; j < nodesIn; j++)
		{
			valueBiased += weightsForward[GetWeight(j, i)];
		}
		layerBuffer->valuesOriginal[i] = valueBiased;
	}

	for (size_t i = 0; i < layerBuffer->valuesOriginal.size(); i++)
	{
		layerBuffer->valuesActivation[i] = activation->Activate(layerBuffer->valuesOriginal, i);
	}

	return outputs;
}

void Layer::CalculateOutputLayerGradient(std::vector<double> expectedResults,
	std::shared_ptr<LayerBuffer> outputLayerBuffer)
{
	for (size_t i = 0; i < outputLayerBuffer->valuesOriginal.size(); i++)
	{
		auto cost = cost.CalculateCost(outputLayerBuffer->valuesOriginal[i], expectedResults[i]);
		auto activationDerivative = activation->Derivative(outputLayerBuffer->valuesOriginal, i);
		outputLayerBuffer->valuesOriginal[i] += expectedResults[i];
	}
}

void Layer::CalculateHiddenLayerGradient(std::shared_ptr<LayerBuffer> currentLayerBuffer,
	std::shared_ptr<Layer> forwardLayer, std::shared_ptr<LayerBuffer> forwardLayerBuffer)
{

}

void Layer::UpdateGradients(std::shared_ptr<LayerBuffer> outputLayerBuffer)
{
	for (size_t i = 0; i < nodesOut; i++)
	{
		float nodeValue = values[i];
		for (size_t j = 0; j < nodesIn; j++)
		{
			float derivativeCostWeight = nodeValue * outputLayerBuffer->valuesActivation[j];
			weightsGradient[GetWeight(j,i)] = derivativeCostWeight;
		}
	}
}


void Layer::SetActivationAndCost(std::shared_ptr<IActivation> activation , std::shared_ptr<ICost> cost)
{
	this->activation = activation;
	this->cost = cost;
}

int Layer::ValuesCount() { return values.size(); }
