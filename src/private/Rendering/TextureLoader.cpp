#include "../include/Rendering/TextureLoader.h"


GLuint TextureLoader::LoadTexture(std::shared_ptr<NeuralDataObject> neuralDataObj)
{
	std::vector<uint8_t> flatArrayTextureData;
	ConvertToArrayTextureData(neuralDataObj->GetFlatObjectPixelsArray(), flatArrayTextureData);

	GLuint texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 28, 28, 0, GL_RGBA, GL_UNSIGNED_BYTE, flatArrayTextureData.data());

	glBindTexture(GL_TEXTURE_2D, 0);
	return texture;
}

void TextureLoader::ConvertToArrayTextureData(std::vector<uint8_t> array , std::vector<uint8_t>& flatArrayToPopulate) {
	for (int y = 0; y < array.size(); ++y) 
	{
		uint8_t pixel_value = array[y] * 255;  // Convert 0-1 range to 0-255 for grayscale
		flatArrayToPopulate.push_back(pixel_value);
		flatArrayToPopulate.push_back(pixel_value);
		flatArrayToPopulate.push_back(pixel_value);
		flatArrayToPopulate.push_back(255);  // Alpha channel (fully opaque)
	}
}