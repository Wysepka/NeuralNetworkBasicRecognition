#pragma once
#include <memory>
#include <iostream>
#include "IActivation.h"

class Sigmoid : public IActivation {
public:
	double Activate(const std::vector<double>& inputs, int index) const override {
		return 1.0 / (1.0 + std::exp(-inputs[index]));
	}

	double Derivative(const std::vector<double>& inputs, int index) const override {
		double a = Activate(inputs, index);
		return a * (1 - a);
	}

	ActivationType GetActivationType() const override {
		return ActivationType::Sigmoid;
	}
};

class TanH : public IActivation {
public:
	double Activate(const std::vector<double>& inputs, int index) const override {
		double e2 = std::exp(2 * inputs[index]);
		return (e2 - 1) / (e2 + 1);
	}

	double Derivative(const std::vector<double>& inputs, int index) const override {
		double e2 = std::exp(2 * inputs[index]);
		double t = (e2 - 1) / (e2 + 1);
		return 1 - t * t;
	}

	ActivationType GetActivationType() const override {
		return ActivationType::TanH;
	}
};

class ReLU : public IActivation {
public:
	double Activate(const std::vector<double>& inputs, int index) const override {
		return std::max(0.0, inputs[index]);
	}

	double Derivative(const std::vector<double>& inputs, int index) const override {
		return (inputs[index] > 0) ? 1.0 : 0.0;
	}

	ActivationType GetActivationType() const override {
		return ActivationType::ReLU;
	}
};

class SiLU : public IActivation {
public:
	double Activate(const std::vector<double>& inputs, int index) const override {
		return inputs[index] / (1.0 + std::exp(-inputs[index]));
	}

	double Derivative(const std::vector<double>& inputs, int index) const override {
		double sig = 1.0 / (1.0 + std::exp(-inputs[index]));
		return inputs[index] * sig * (1 - sig) + sig;
	}

	ActivationType GetActivationType() const override {
		return ActivationType::SiLU;
	}
};

class Softmax : public IActivation {
public:
	double Activate(const std::vector<double>& inputs, int index) const override {
		double expSum = 0;
		for (double input : inputs) {
			expSum += std::exp(input);
		}
		return std::exp(inputs[index]) / expSum;
	}

	double Derivative(const std::vector<double>& inputs, int index) const override {
		double expSum = 0;
		for (double input : inputs) {
			expSum += std::exp(input);
		}
		double ex = std::exp(inputs[index]);
		return (ex * expSum - ex * ex) / (expSum * expSum);
	}

	ActivationType GetActivationType() const override {
		return ActivationType::Softmax;
	}
};

class Activation
{
	static std::unique_ptr<IActivation> GetActivationFromType(ActivationType type) {
		switch (type) {
		case ActivationType::Sigmoid:
			return std::make_unique<class Sigmoid>();
		case ActivationType::TanH:
			return std::make_unique<class TanH>();
		case ActivationType::ReLU:
			return std::make_unique<class ReLU>();
		case ActivationType::SiLU:
			return std::make_unique<class SiLU>();
		case ActivationType::Softmax:
			return std::make_unique<class Softmax>();
		default:
			std::cout << "Unhandled activation type" << std::endl;
			return std::make_unique<class Sigmoid>();
		}
	}
};
