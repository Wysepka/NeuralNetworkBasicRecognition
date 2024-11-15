#pragma once
#include <memory>
#include "Config/ProjectConfig.h"
#include "Loader/FileLoader.h"
#include "Log/Logger.h"
#include "Rendering/RenderingSystem.h"
#include "Event/MessageBus.h"
#include "Neural/NeuralNetworkController.h"

class Engine 
{
private:
	std::shared_ptr<FileLoader> fileLoader;
	std::shared_ptr<Logger> logger;
	std::shared_ptr<RenderingSystem> renderingSystem;
	std::shared_ptr<MessageBus> messageBus;
	std::shared_ptr<NeuralNetworkController> neuralNetworkController;
public:
	void InitializeNeuralNetworkProcess();

	void Initialize();
	void ProcessMainLoop();
	void Dispose();
};