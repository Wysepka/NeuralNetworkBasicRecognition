#include "Neural/Cost.h"

std::unique_ptr<ICost> Cost::GetCostFromType(CostType type)
{
    switch (type)
    {
        case MeanSquareError:
            return std::make_unique<class MeanSquaredError>();
        case CrossEntropy:
            return std::make_unique<class CrossEntropy>();
        default:
            std::cerr << "Unhandled cost type" << std::endl;
        return std::make_unique<MeanSquaredError>();
    }
}