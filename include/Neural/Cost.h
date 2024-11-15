#pragma once
#include <memory>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <iostream>
#include "ICost.h"

class Cost
{
public:
	enum CostType
	{
		MeanSquareError,
		CrossEntropy
	};

	static std::unique_ptr<ICost> GetCostFromType(CostType type);
};

class MeanSquaredError : public ICost
{
public:
	double CostFunction(const std::vector<double>& predictedOutputs, const std::vector<double>& expectedOutputs) override
	{
		double cost = 0;
		for (size_t i = 0; i < predictedOutputs.size(); ++i)
		{
			double error = predictedOutputs[i] - expectedOutputs[i];
			cost += error * error;
		}
		return 0.5 * cost;
	}

	double CostDerivative(double predictedOutput, double expectedOutput) override
	{
		return predictedOutput - expectedOutput;
	}
};

class CrossEntropy : public ICost
{
public:
	double CostFunction(const std::vector<double>& predictedOutputs, const std::vector<double>& expectedOutputs) override
	{
		double cost = 0;
		for (size_t i = 0; i < predictedOutputs.size(); ++i)
		{
			double x = predictedOutputs[i];
			double y = expectedOutputs[i];
			double v = (y == 1) ? -std::log(x) : -std::log(1 - x);
			if (!std::isnan(v))
				cost += v;
		}
		return cost;
	}

	double CostDerivative(double predictedOutput, double expectedOutput) override
	{
		double x = predictedOutput;
		double y = expectedOutput;
		if (x == 0 || x == 1)
		{
			return 0;
		}
		return (-x + y) / (x * (x - 1));
	}
};

