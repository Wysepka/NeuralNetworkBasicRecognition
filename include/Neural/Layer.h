#pragma once
#include <vector>
#include <cmath>    // For sqrt, log, cos, and M_PI
#include <random>   // For std::mt19937 and std::uniform_real_distribution
#include <chrono>
#include <stdexcept>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum LayerType : uint8_t
{
	Input = 0,
	Hidden = 1,
	Output = 2,
};

class Layer
{
private:
	std::vector<double> values;
	std::vector<double> weightsForward;
	std::vector<double> biases;

	std::vector<double> weightsGradient;
	std::vector<double> biasesGradient;

	LayerType layerType;

	std::vector<double> weights;

	// Function to initialize weights with random values following a normal distribution
	void InitializeRandomWeights(std::mt19937& rng);
	double RandomInNormalDistribution(std::mt19937& rng, double mean, double standardDeviation);

public:
	Layer(unsigned int nodes, unsigned int nodesOut , LayerType layerType) 
		: values(nodes) , weightsForward(nodes * nodesOut) , biases(values) , weightsGradient(nodes * nodesOut) , biasesGradient(nodes) , layerType(layerType)
	{
		std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
		InitializeRandomWeights(rng);
	}

	std::vector<double> CalculateValues(std::vector<double> inputs);
};