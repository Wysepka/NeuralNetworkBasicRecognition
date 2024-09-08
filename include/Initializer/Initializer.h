#pragma once
#include <memory>
#include "Config/ProjectConfig.h"
#include "Loader/FileLoader.h"
#include "Log/Logger.h"

class Initializer 
{
private:
	std::shared_ptr<FileLoader> fileLoader;
	std::shared_ptr<Logger> logger;
public:
	void Initialize();
};