#include "Log/Logger.h"

void Logger::LogDataFile(const std::shared_ptr<const NeuralDataFile> dataFile)
{
	auto objects = dataFile->GetNeuralDataObjects();
	for (size_t i = 0; i < objects.size(); i++)
	{
		int label = objects[i]->GetLabel();
		std::cout << "======== LABEL: " << label << "| ITERATION: " << i << "========\n";
		auto pixels = objects[i]->GetFlatObjectPixelsArray();
		for (size_t j = 0; j < pixels.size(); j++)
		{
			if ((j + 1) % objects[i]->GetXDim() == 0)
			{
				std::cout << '\n';
			}
			std::cout << pixels[j];
		}
		std::cout << '\n';
	}
}