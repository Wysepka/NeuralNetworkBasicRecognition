#pragma once
#include <vector>
#include <cmath>    // For sqrt, log, cos, and M_PI
#include <random>   // For std::mt19937 and std::uniform_real_distribution
#include <chrono>
#include <memory>
#include <stdexcept>
#include <Neural/LayerBuffer.h>
#include "IActivation.h"


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

	LayerType layerType;

	int nodesIn;
	int nodesOut;

	std::shared_ptr<IActivation> activation;

	// Function to initialize weights with random values following a normal distribution
	void InitializeRandomWeights(std::mt19937& rng);
	double RandomInNormalDistribution(std::mt19937& rng, double mean, double standardDeviation);
	double GetWeight(unsigned int nodeId, unsigned int nodeOut);

public:
	Layer(unsigned int nodes, unsigned int nodesOut , LayerType layerType) 
		: nodesIn(nodes) , nodesOut(nodesOut), values(nodes) , weightsForward(nodes * nodesOut) , biases(nodesOut) , weightsGradient(nodes * nodesOut) , biasesGradient(nodes) , layerType(layerType)
	{
		std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
		InitializeRandomWeights(rng);
	}

	void SetActivation(std::shared_ptr<IActivation> activation);
	 
	int ValuesCount();

	std::vector<double> CalculateValues(std::vector<double> inputs, std::shared_ptr<LayerBuffer> layerBuffer);
	void CalculateOutputLayerResults(std::vector<double> expectedResults, std::shared_ptr<LayerBuffer> outputLayerBuffer);
};