#pragma once
#include <stdio.h>
//#include "GLFW/glfw3.h"
#include "../lib/glad/include/glad/glad.h"
#include "../lib/glfw/include/GLFW/glfw3.h"
#include "../lib/imgui/imgui.h"
#include "../lib/imgui/backends/imgui_impl_opengl3.h"
#include "../lib/imgui/backends/imgui_impl_glfw.h"
#include "Event/MessageBus.h"
#include "Rendering/TextureLoader.h"

class RenderingSystem
{
private:
	std::shared_ptr<TextureLoader> textureLoader;
	bool showDemoWindow;
	bool showAnotherWindow;
	GLFWwindow* window;
	ImVec4 clear_color;
	ImGuiIO* io;
	std::shared_ptr<MessageBus> messageBus;


	uint32_t currentTextureIndex;
	int textureWidth, textureHeight;
	std::vector<GLuint> textures;

	void OnFileLoadEnded(const std::shared_ptr<EndLoadingFile>& message);

	void DisplayLoadedTextureData();

public:
	RenderingSystem(bool demoWindow, bool anotherWindow, ImGuiIO* ioRef , std::shared_ptr<MessageBus> messageBus)
		: showDemoWindow(demoWindow), showAnotherWindow(anotherWindow), io(ioRef), window(nullptr) , messageBus(messageBus) {}

	int Initialize();
	void ProcessMainRenderingLoop(bool& shouldBeRendering, int& retFlag);
    void DisposeRendering();
	static void glfw_error_callback(int error, const char* description);
};