#pragma once
#include <memory>
#include "Config/ProjectConfig.h"
#include "Loader/FileLoader.h"
#include "Log/Logger.h"
#include "Rendering/RenderingSystem.h"

class Engine 
{
private:
	std::shared_ptr<FileLoader> fileLoader;
	std::shared_ptr<Logger> logger;
	std::shared_ptr<RenderingSystem> renderingSystem;
public:
	void Initialize();
	void ProcessMainLoop();
	void Dispose();
};