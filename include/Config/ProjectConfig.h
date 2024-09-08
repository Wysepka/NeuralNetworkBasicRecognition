#pragma once 
#include <array>
#include "Config/FileLoadConfig.h"

class ProjectConfig 
{
public:
	static std::array<std::shared_ptr<FileLoadConfig>, 1> LoadTypesQueue;
	
};