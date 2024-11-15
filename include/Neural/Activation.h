#pragma once
#include <memory>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
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
        return ActivationType::ActivationType_Sigmoid;
    }
};

class TanH : public IActivation {
public:
    double Activate(const std::vector<double>& inputs, int index) const override {
        return std::tanh(inputs[index]);
    }

    double Derivative(const std::vector<double>& inputs, int index) const override {
        double t = std::tanh(inputs[index]);
        return 1 - t * t;
    }

    ActivationType GetActivationType() const override {
        return ActivationType::ActivationType_TanH;
    }
};

class ReLU : public IActivation {
public:
    double Activate(const std::vector<double>& inputs, int index) const override {
        return (std::max)(0.0, inputs[index]);
    }

    double Derivative(const std::vector<double>& inputs, int index) const override {
        return (inputs[index] > 0) ? 1.0 : 0.0;
    }

    ActivationType GetActivationType() const override {
        return ActivationType::ActivationType_ReLU;
    }
};

class SiLU : public IActivation {
public:
    double Activate(const std::vector<double>& inputs, int index) const override {
        return inputs[index] / (1.0 + std::exp(-inputs[index]));
    }

    double Derivative(const std::vector<double>& inputs, int index) const override {
        double sig = 1.0 / (1.0 + std::exp(-inputs[index]));
        return sig * (1 + inputs[index] * (1 - sig));
    }

    ActivationType GetActivationType() const override {
        return ActivationType::ActivationType_SiLU;
    }
};

class Softmax : public IActivation {
public:
    double Activate(const std::vector<double>& inputs, int index) const override {
        // Subtract the max value for numerical stability
        double maxInput = *std::max_element(inputs.begin(), inputs.end());
        double expSum = 0.0;
        for (double input : inputs) {
            expSum += std::exp(input - maxInput);
        }
        return std::exp(inputs[index] - maxInput) / expSum;
    }

    double Derivative(const std::vector<double>& inputs, int index) const override {
        double output = Activate(inputs, index);
        return output * (1 - output);
    }

    ActivationType GetActivationType() const override {
        return ActivationType::ActivationType_Softmax;
    }
};

class Activation {
public:
    static std::unique_ptr<IActivation> GetActivationFromType(ActivationType type) {
        switch (type) {
            case ActivationType::ActivationType_Sigmoid:
                return std::make_unique<Sigmoid>();
            case ActivationType::ActivationType_TanH:
                return std::make_unique<TanH>();
            case ActivationType::ActivationType_ReLU:
                return std::make_unique<ReLU>();
            case ActivationType::ActivationType_SiLU:
                return std::make_unique<SiLU>();
            case ActivationType::ActivationType_Softmax:
                return std::make_unique<Softmax>();
            default:
                std::cerr << "Unhandled activation type" << std::endl;
                return nullptr;
        }
    }
};