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
	layerBuffer->valuesOriginal = inputs;
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
		layerBuffer->valuesCalculated[i] = valueBiased;
	}

	for (size_t i = 0; i < layerBuffer->valuesCalculated.size(); i++)
	{
		layerBuffer->valuesActivation[i] = activation->Activate(layerBuffer->valuesCalculated, i);
	}

	return outputs;
}

void Layer::CalculateOutputLayerGradient(std::vector<double> expectedResults,
	std::shared_ptr<LayerBuffer> outputLayerBuffer)
{
	for (size_t i = 0; i < outputLayerBuffer->valuesOriginal.size(); i++)
	{
		auto costValue = cost->CostFunction(outputLayerBuffer->valuesActivation, expectedResults);
		auto activationDerivative = activation->Derivative(outputLayerBuffer->valuesCalculated, i);
		outputLayerBuffer->valuesGradient[i] = costValue * activationDerivative;
	}
}

void Layer::CalculateHiddenLayerGradient(std::shared_ptr<LayerBuffer> currentLayerBuffer,
	std::shared_ptr<Layer> forwardLayer, std::shared_ptr<LayerBuffer> forwardLayerBuffer)
{
	for (size_t i = 0; i < nodesOut; i++)
	{
		double newNodeValue = 0.f;
		for (size_t j = 0; j < nodesIn; j++)
		{
			auto forwardWeight = forwardLayer->weightsForward[GetWeight(j, i)];
			newNodeValue += forwardWeight * forwardLayerBuffer->valuesGradient[j];
		}
		newNodeValue += activation->Derivative(currentLayerBuffer->valuesCalculated, i);
		currentLayerBuffer->valuesGradient[i] = newNodeValue;
	}
}

void Layer::UpdateGradients(std::shared_ptr<LayerBuffer> currentLayerBuffer)
{
	for (size_t i = 0; i < nodesOut; i++)
	{
		float nodeValue = currentLayerBuffer->valuesActivation[i];
		for (size_t j = 0; j < nodesIn; j++)
		{
			float derivativeCostWeight = nodeValue * currentLayerBuffer->valuesOriginal[j];
			weightsGradient[GetWeight(j,i)] += derivativeCostWeight;
		}
	}
	for (size_t i = 0; i < biases.size(); i++)
	{
		double derivativeCostBias = 1 * currentLayerBuffer->valuesActivation[i];
		biasesGradient[i] += derivativeCostBias;
	}
}

void Layer::ApplyGradients(double learnRate, double decayFactor, double momentum)
{
	float decayValue = (1 - decayFactor * learnRate);

	for (size_t i = 0; i < weightsForward.size(); i++)
	{
		double velocity = weightVelocity[i] * momentum - weightsGradient[i] * learnRate;
		weightVelocity[i] = velocity;
		weightsForward[i] = weightsForward[i] + weightVelocity[i] * decayValue;
		weightsGradient[i] = 0;
	}

	for (size_t i = 0; i < biases.size(); i++) {
		double velocity = biasVelocity[i] * momentum - biases[i] * learnRate;
		biasVelocity[i] = velocity;
		biases[i] = biases[i] + biasVelocity[i] * decayValue;
		biasesGradient[i] = 0;
	}
}


void Layer::SetActivationAndCost(std::shared_ptr<IActivation> activation , std::shared_ptr<ICost> cost)
{
	this->activation = activation;
	this->cost = cost;
}

int Layer::ValuesCount() { return values.size(); }
