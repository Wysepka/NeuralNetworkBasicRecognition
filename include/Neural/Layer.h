#pragma once
#include <vector>
#include <cmath>    // For sqrt, log, cos, and M_PI
#include <random>   // For std::mt19937 and std::uniform_real_distribution
#include <chrono>
#include <memory>
#include <Neural/LayerBuffer.h>
#include "IActivation.h"
#include "ICost.h"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class Layer {
private:
	std::vector<double> values;
	std::vector<double> weightsForward;
	std::vector<double> biases;

	std::vector<double> weightsGradient;
	std::vector<double> biasesGradient;

	std::vector<double> weightVelocity;
	std::vector<double> biasVelocity;

	LayerType layerType;

	int nodesIn;
	int nodesOut;

	std::shared_ptr<IActivation> activation;
	std::shared_ptr<ICost> cost;

	// Function to initialize weights with random values following a normal distribution
	void InitializeRandomWeights(std::mt19937& rng);
	double RandomInNormalDistribution(std::mt19937& rng, double mean, double standardDeviation);
	double GetWeight(unsigned int nodeId, unsigned int nodeOut);

public:
	Layer(unsigned int nodesIn, unsigned int nodesOut , LayerType layerType)
		: nodesIn(nodesIn) , nodesOut(nodesOut), values(nodesIn) , weightsForward(nodesIn * nodesOut) , biases(nodesOut) , weightsGradient(nodesIn * nodesOut) , biasesGradient(nodesIn) , weightVelocity(nodesIn*nodesOut) , biasVelocity(nodesIn) , layerType(layerType)
	{
		std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
		InitializeRandomWeights(rng);
	}

	void SetActivationAndCost(std::shared_ptr<IActivation> activation , std::shared_ptr<ICost> cost);
	 
	int ValuesCount();

	std::vector<double> CalculateValues(std::vector<double> inputs, std::shared_ptr<LayerBuffer> layerBuffer);
	void CalculateOutputLayerGradient(std::vector<double> expectedResults, std::shared_ptr<LayerBuffer> outputLayerBuffer);
	void CalculateHiddenLayerGradient(std::shared_ptr<LayerBuffer> currentLayerBuffer, std::shared_ptr<Layer> forwardLayer, std::shared_ptr<LayerBuffer> forwardLayerBuffer);

	void UpdateGradients(std::shared_ptr<LayerBuffer> currentLayerBuffer);
	void ApplyGradients(double learnRate, double decayFactor, double velocity);
};