#pragma once
#include <stdio.h>
//#include "GLFW/glfw3.h"
#include "../lib/glfw/include/GLFW/glfw3.h"
//#include "../lib/glad/include/glad/glad.h"
#include "../lib/imgui/imgui.h"
#include "../lib/imgui/backends/imgui_impl_opengl3.h"
#include "../lib/imgui/backends/imgui_impl_glfw.h"

class RenderingSystem
{
private:
	bool showDemoWindow;
	bool showAnotherWindow;
	GLFWwindow* window;
	ImVec4 clear_color;
	ImGuiIO* io;

public:
	RenderingSystem(bool demoWindow, bool anotherWindow, ImGuiIO* ioRef)
		: showDemoWindow(demoWindow), showAnotherWindow(anotherWindow), io(ioRef), window(nullptr) {}

	int Initialize();
	void ProcessMainRenderingLoop(bool& shouldBeRendering, int& retFlag);
    void DisposeRendering();
	static void glfw_error_callback(int error, const char* description);
};