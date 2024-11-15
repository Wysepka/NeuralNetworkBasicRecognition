#include "Engine/Engine.h"
#include "../lib/imgui/backends/imgui_impl_glfw.h"
#include "../lib/imgui/backends/imgui_impl_opengl3.h"
#include "../lib/imgui/imgui.h"

void Engine::Initialize()
{
	messageBus = std::make_shared<MessageBus>();
	fileLoader = std::make_shared<FileLoader>(messageBus);
	logger = std::make_shared<Logger>();
	neuralNetworkController = std::make_shared<NeuralNetworkController>();

	renderingSystem = std::make_shared<RenderingSystem>(true, true, nullptr , messageBus);
	renderingSystem->Initialize();
	InitializeNeuralNetworkProcess();
	//ImGui::CreateContext
}

void Engine::InitializeNeuralNetworkProcess() {
	auto loadConfigs = ProjectConfig::LoadTypesQueue;
	for (size_t i = 0; i < loadConfigs.size(); i++)
	{
		auto data = fileLoader->LoadFile(loadConfigs[i]);
		//logger->LogDataFile(data);
		neuralNetworkController->Initialize(data);
	}
}

void Engine::ProcessMainLoop()
{
	bool bRenderingSystemRunning = true;
	bool bNeuralNetworkSystemRunning = true;
	while (bRenderingSystemRunning || bNeuralNetworkSystemRunning)
	{
		int retFlag;
		renderingSystem->ProcessMainRenderingLoop(bRenderingSystemRunning, retFlag);
		neuralNetworkController->Run(bNeuralNetworkSystemRunning);
	}
	renderingSystem->DisposeRendering();
}

void Engine::Dispose()
{

}
