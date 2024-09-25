#pragma once
#include <vector>

class ICost
{
public:
	virtual double CostFunction(const std::vector<double>& predictedOutputs, const std::vector<double>& expectedOutputs) = 0;
	virtual double CostDerivative(double predictedOutput, double expectedOutput) = 0;
	virtual ~ICost() {}
};