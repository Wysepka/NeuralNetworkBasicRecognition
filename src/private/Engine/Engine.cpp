#include "Engine/Engine.h"
#include "../lib/imgui/backends/imgui_impl_glfw.h"
#include "../lib/imgui/backends/imgui_impl_opengl3.h"
#include "../lib/imgui/imgui.h"

void Engine::Initialize() 
{
	messageBus = std::make_shared<MessageBus>();
	fileLoader = std::make_shared<FileLoader>(messageBus);
	logger = std::make_shared<Logger>();

	renderingSystem = std::make_shared<RenderingSystem>(true, true, nullptr , messageBus);
	renderingSystem->Initialize();
	auto loadConfigs = ProjectConfig::LoadTypesQueue;
	for (size_t i = 0; i < loadConfigs.size(); i++)
	{
		auto data = fileLoader->LoadFile(loadConfigs[i]);
		//logger->LogDataFile(data);
	}
	//ImGui::CreateContext
}

void Engine::ProcessMainLoop()
{
	bool bRenderingSystemRunning = true;
	while (bRenderingSystemRunning) 
	{
		int retFlag;
		renderingSystem->ProcessMainRenderingLoop(bRenderingSystemRunning, retFlag);
	}
	renderingSystem->DisposeRendering();
}

void Engine::Dispose()
{

}
