#pragma once
#include <string>
#include <cstdint>
#include "Data/NeuralDataFile.h"

enum LoadPathType : uint8_t
{
	InvalidLoadPath = 0,
	Single = 1,
	Multiple_Labels_Images = 2,
};

struct FileLoadConfig {
public:
	virtual const std::string FileDirectoryPath() = 0;
	virtual const std::vector<std::string> FileDirectoryPaths() = 0;
	virtual const NeuralDataObject_Type GetNeuralType()
	{
		return NeuralDataObject_Type::Invalid;
	}
	virtual LoadPathType GetLoadPathType() 
	{
		return LoadPathType::InvalidLoadPath;
	}
};

struct FileLoadConfig_MNIST : FileLoadConfig
{
private:
	NeuralDataObject_Type type;
	const LoadPathType PathType = LoadPathType::Multiple_Labels_Images;
public:

	virtual const std::string FileDirectoryPath() override 
	{
		return "INVALID Use FileDirectoryPath(s) due its multiple data files!";
	}

	/// <summary>
	/// Images = 0 vector element index
	/// Labels = 1 vector element index
	/// </summary>
	/// <returns></returns>
	virtual const std::vector<std::string> FileDirectoryPaths() override 
	{
		return std::vector<std::string>
		{
			std::string("D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/Data/Kaggle_Mnist/train-images.idx3-ubyte"),
			std::string("D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/Data/Kaggle_Mnist/train-labels.idx1-ubyte")
		};
	}

	virtual LoadPathType GetLoadPathType() override 
	{
		return PathType;
	}

	virtual const NeuralDataObject_Type GetNeuralType() override 
	{
		return NeuralDataObject_Type::MNIST_Digit;
	}
};