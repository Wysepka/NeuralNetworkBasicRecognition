#include "Loader/FileLoader.h"

std::shared_ptr<NeuralDataFile> FileLoader::LoadFile(std::shared_ptr<FileLoadConfig> config) 
{
	std::shared_ptr<NeuralDataFile> neuralDataFile = NeuralDataFile::CreateInstanceOfNeuralDataFile(config->GetNeuralType());

	LoadPathType pathType = config->GetLoadPathType();
	if (pathType == LoadPathType::Multiple_Labels_Images) 
	{
		auto paths = config->FileDirectoryPaths();
		std::string images_path = paths[0];
		std::string labels_path = paths[1];

		int number_of_test_images, test_rows, test_cols;
		std::vector<std::vector<uint8_t>> test_images = ReadImages(images_path, number_of_test_images, test_rows, test_cols);
		std::vector<uint8_t> test_labels = ReadLabels(labels_path);

		if (test_images.size() != test_labels.size()) {
			std::ostringstream error;
			error << "Images size vector length is different, than Images Labels ! For Files Dir [0]: " <<
				images_path << " ||| [1]: " << labels_path << "\n";
			throw std::runtime_error(error.str());
			return neuralDataFile;
		}

		neuralDataFile->Initialize(test_images.size());

		for (size_t i = 0; i < test_images.size(); i++)
		{
			std::shared_ptr<NeuralDataObject> dataObject = std::make_shared<NeuralDataObject>();
			dataObject->SetFlatObjectsPixelsArray(std::make_shared<std::vector<uint8_t>>(test_images[i]));
			dataObject->SetLabel(test_labels[i]);
			dataObject->SetDimensions(NeuralDataFile_MNIST_Digits::X_DIMENSION, NeuralDataFile_MNIST_Digits::Y_DIMENSION);
			neuralDataFile->AddDataObject(dataObject);
		}
	}
	return neuralDataFile;
}



uint32_t FileLoader::SwapEndian(uint32_t val) {
	return ((val >> 24) & 0x000000FF) |
		((val >> 8) & 0x0000FF00) |
		((val << 8) & 0x00FF0000) |
		((val << 24) & 0xFF000000);
}

std::vector<uint8_t> FileLoader::ReadLabels(const std::string& file_path) {
	std::ifstream file(file_path, std::ios::binary);
	assert(file.is_open());

	uint32_t magic_number = 0;
	uint32_t number_of_labels = 0;

	// Read and convert the magic number and number of labels
	file.read(reinterpret_cast<char*>(&magic_number), 4);
	magic_number = SwapEndian(magic_number);  // Convert from big-endian to little-endian

	file.read(reinterpret_cast<char*>(&number_of_labels), 4);
	number_of_labels = SwapEndian(number_of_labels);  // Convert from big-endian to little-endian

	// Read the labels (as unsigned 8-bit integers)
	std::vector<uint8_t> labels(number_of_labels);
	file.read(reinterpret_cast<char*>(labels.data()), number_of_labels);

	return labels;
}

std::vector<std::vector<uint8_t>> FileLoader::ReadImages(const std::string& file_path, int& number_of_images, int& rows, int& cols) {
	std::ifstream file(file_path, std::ios::binary);
	assert(file.is_open());

	uint32_t magic_number = 0;

	// Read and convert the magic number, number of images, rows, and columns
	file.read(reinterpret_cast<char*>(&magic_number), 4);
	magic_number = SwapEndian(magic_number);

	file.read(reinterpret_cast<char*>(&number_of_images), 4);
	number_of_images = SwapEndian(number_of_images);

	file.read(reinterpret_cast<char*>(&rows), 4);
	rows = SwapEndian(rows);

	file.read(reinterpret_cast<char*>(&cols), 4);
	cols = SwapEndian(cols);

	// Read the image data
	std::vector<std::vector<uint8_t>> images(number_of_images, std::vector<uint8_t>(rows * cols));
	for (int i = 0; i < number_of_images; ++i) {
		file.read(reinterpret_cast<char*>(images[i].data()), rows * cols);
	}

	return images;
}