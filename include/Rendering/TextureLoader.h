#pragma once
#include <vector>
#include <memory>
#include "../lib/glad/include/glad/glad.h"
#include "../lib/glfw/include/GLFW/glfw3.h"
#include "../include/Data/NeuralDataFile.h"

class TextureLoader
{
private:
	void ConvertToArrayTextureData(std::vector<uint8_t> array, std::vector<uint8_t>& flatArrayToPopulate);
public:
	GLuint LoadTexture(std::shared_ptr<NeuralDataObject> neuralDataObj);

};