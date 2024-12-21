#include "Neural/Layer.h"

void Layer::InitializeRandomWeights(std::mt19937& rng)
{
	for (size_t i = 0; i < weightsBackwards.size(); i++)
	{
		weightsBackwards[i] = RandomInNormalDistribution(rng, 0.0, 1.0) / std::sqrt(values.size());
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
	return weightsBackwards[index];
}

int Layer::GetFlatWeightIndex(unsigned int nodeIn, unsigned int nodeOut)
{
	return nodesIn * nodeOut + nodeIn;
}


std::vector<double> Layer::CalculateValues(std::vector<double> inputs, std::shared_ptr<LayerBuffer> layerBuffer)
{
	layerBuffer->valuesOriginal = inputs;
	std::vector<double> outputs;
	if (inputs.size() != nodesIn)
	{
		throw std::runtime_error("Inputs Size is different than NodesIn size in input Layer !");
		return outputs;
	}
	for (size_t i = 0; i < nodesOut; i++)
	{
		double valueBiased = biases[i];
		for (size_t j = 0; j < nodesIn; j++)
		{
			double weightValue = GetWeight(j, i);
			valueBiased += weightValue;
			if(valueBiased > 10)
			{
				auto s = 's';
			}
		}
		layerBuffer->valuesCalculated[i] = valueBiased;
	}

	for (size_t i = 0; i < layerBuffer->valuesCalculated.size(); i++)
	{
		layerBuffer->valuesActivation[i] = activation->Activate(layerBuffer->valuesCalculated, i);
	}

	return layerBuffer->valuesActivation;
}

void Layer::CalculateOutputLayerGradient(std::vector<double> expectedResults,
	std::shared_ptr<LayerBuffer> outputLayerBuffer)
{
	for (size_t i = 0; i < outputLayerBuffer->valuesGradient.size(); i++)
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
		for (size_t j = 0; j < forwardLayer->NodesOut(); j++)
		{
			auto forwardWeight = forwardLayer->GetWeight(i, j);
			newNodeValue += forwardWeight * forwardLayerBuffer->valuesGradient[j];
		}
		newNodeValue *= activation->Derivative(currentLayerBuffer->valuesCalculated, i);
		currentLayerBuffer->valuesGradient[i] = newNodeValue;
	}
}

void Layer::UpdateGradients(std::shared_ptr<LayerBuffer> currentLayerBuffer)
{
	for (size_t i = 0; i < nodesOut; i++)
	{
		double nodeValue = currentLayerBuffer->valuesGradient[i];
		for (size_t j = 0; j < nodesIn; j++)
		{
			double derivativeCostWeight = nodeValue * currentLayerBuffer->valuesOriginal[j];
			int flatWeightIndex = GetFlatWeightIndex(j,i);
			weightsGradient[flatWeightIndex] += derivativeCostWeight;
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
	{
		std::lock_guard<std::mutex> lock(applyGradientMutex);

		float decayValue = (1 - decayFactor * learnRate);

		for (size_t i = 0; i < weightsBackwards.size(); i++)
		{
			double velocity = weightVelocity[i] * momentum - weightsGradient[i] * learnRate;
			weightVelocity[i] = velocity;
			weightsBackwards[i] = weightsBackwards[i] + weightVelocity[i] * decayValue;
			weightsGradient[i] = 0;
		}

		for (size_t i = 0; i < biases.size(); i++) {
			double velocity = biasVelocity[i] * momentum - biases[i] * learnRate;
			biasVelocity[i] = velocity;
			biases[i] += velocity;
			biasesGradient[i] = 0;
		}
	}
}


void Layer::SetActivationAndCost(std::shared_ptr<IActivation> activation , std::shared_ptr<ICost> cost)
{
	this->activation = activation;
	this->cost = cost;
}

int Layer::ValuesCount() { return values.size(); }

int Layer::NodesIn() {
	return nodesIn;
}

int Layer::NodesOut() {
	return nodesOut;
}
