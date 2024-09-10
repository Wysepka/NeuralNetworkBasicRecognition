#pragma once
#include "Engine/Engine.h"

int main() {
	Engine initializer;
	initializer.Initialize();
	initializer.ProcessMainLoop();
	initializer.Dispose();

	return 0;
}