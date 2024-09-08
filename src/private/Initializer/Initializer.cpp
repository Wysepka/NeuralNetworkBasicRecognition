#include "Initializer/Initializer.h"


void Initializer::Initialize() 
{
	fileLoader = std::make_shared<FileLoader>();
	logger = std::make_shared<Logger>();
	auto loadConfigs = ProjectConfig::LoadTypesQueue;
	for (size_t i = 0; i < loadConfigs.size(); i++)
	{
		auto data = fileLoader->LoadFile(loadConfigs[i]);
		logger->LogDataFile(data);
	}
}
