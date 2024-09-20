#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <sstream>
#include <cassert>
#include <thread>
#include "../lib/MNISTReader/include/mnist/mnist_reader_less.hpp"
#include "Settings/FileLoadSettings.h"
#include "Config/FileLoadConfig.h"
#include "Data/NeuralDataFile.h"
#include "Event/MessageBus.h"

class FileLoader 
{
private:
	std::shared_ptr<MessageBus> messageBus;
	void LoadFile_Internal(std::shared_ptr<NeuralDataFile> dataFile, std::shared_ptr<FileLoadConfig> config);

public:
	FileLoader(std::shared_ptr<MessageBus> messageBus) : messageBus(messageBus)
	{

	};
	std::shared_ptr<NeuralDataFile> LoadFile(std::shared_ptr<FileLoadConfig> config);

	std::vector<std::vector<uint8_t>> ReadImages(const std::string& file_path, int& number_of_images, int& rows, int& cols);
	std::vector<uint8_t> ReadLabels(const std::string& file_path);
	uint32_t SwapEndian(uint32_t val);
};