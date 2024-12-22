#pragma once
#include <vector>
#include <cmath>    // For sqrt, log, cos, and M_PI
#include <random>   // For std::mt19937 and std::uniform_real_distribution
#include <chrono>
#include <memory>
#include <stdexcept>
#include <mutex>
#include <Neural/LayerBuffer.h>
#include "IActivation.h"
#include "ICost.h"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class Layer {
private:
	std::vector<double> values;
	std::vector<double> weightsBackwards;
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

	std::mutex applyGradientMutex;

	// Function to initialize weights with random values following a normal distribution
	void InitializeRandomWeights(std::mt19937& rng);
	void InitializeRandomBiases(std::mt19937& rng);

	std::vector<double> InitializeBiases(int nOut, double minValue = 0.0, double maxValue = 0.01);
	std::vector<double> InitializeWeightsXavier(int nIn, int nOut, double scale = 1.0);
	std::vector<double> InitializeWeightsHe(int nIn, int nOut, double scale = 1.0);

	double RandomInNormalDistribution(std::mt19937& rng, double mean, double standardDeviation);

public:
	Layer(unsigned int nodesIn, unsigned int nodesOut , LayerType layerType)
		: nodesIn(nodesIn) , nodesOut(nodesOut), values(nodesOut) , weightsBackwards(nodesIn * nodesOut) , biases(nodesOut) , weightsGradient(nodesIn * nodesOut) , biasesGradient(nodesIn) , weightVelocity(nodesIn*nodesOut) , biasVelocity(nodesIn) , layerType(layerType)
	{
		//std::mt19937 rngWeights(std::chrono::steady_clock::now().time_since_epoch().count());
		//std::mt19937 rngBiases(std::chrono::steady_clock::now().time_since_epoch().count());
		//InitializeRandomWeights(rngWeights);
		//InitializeRandomBiases(rngBiases);
		biases = InitializeBiases(nodesOut , 0 , 0.01);
		weightsBackwards = InitializeWeightsXavier(nodesIn , nodesOut , 0.05f);
	}

	void SetActivationAndCost(std::shared_ptr<IActivation> activation , std::shared_ptr<ICost> cost);
	 
	int ValuesCount();
	int NodesIn();
	int NodesOut();

	std::vector<double> CalculateValues(std::vector<double> inputs, std::shared_ptr<LayerBuffer> layerBuffer);
	void CalculateOutputLayerGradient(std::vector<double> expectedResults, std::shared_ptr<LayerBuffer> outputLayerBuffer);
	void CalculateHiddenLayerGradient(std::shared_ptr<LayerBuffer> currentLayerBuffer, std::shared_ptr<Layer> forwardLayer, std::shared_ptr<LayerBuffer> forwardLayerBuffer);

	void UpdateGradients(std::shared_ptr<LayerBuffer> currentLayerBuffer);
	void ApplyGradients(double learnRate, double decayFactor, double velocity);

	double GetWeight(unsigned int nodeId, unsigned int nodeOut);
	int GetFlatWeightIndex(unsigned int nodeIn, unsigned int nodeOut);
};