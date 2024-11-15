#pragma once
#include <vector>
#include <memory>


enum ActivationType : uint8_t
{
	ActivationType_Invalid = 0,
	ActivationType_Sigmoid = 1,
	ActivationType_TanH = 2,
	ActivationType_ReLU = 3,
	ActivationType_SiLU = 4,
	ActivationType_Softmax = 5,
};

class IActivation {
public:
	virtual double Activate(const std::vector<double>& inputs, int index) const = 0;
	virtual double Derivative(const std::vector<double>& inputs, int index) const = 0;
	virtual ActivationType GetActivationType() const = 0;
	virtual ~IActivation() {}
};