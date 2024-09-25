#pragma once
#include <vector>
#include <memory>


enum ActivationType : uint8_t
{
	Invalid = 0,
	Sigmoid = 1,
	TanH = 2,
	ReLU = 3,
	SiLU = 4,
	Softmax = 5,
};

class IActivation {
public:
	virtual double Activate(const std::vector<double>& inputs, int index) const = 0;
	virtual double Derivative(const std::vector<double>& inputs, int index) const = 0;
	virtual ActivationType GetActivationType() const = 0;
	virtual ~IActivation() {}
};