#pragma once
#include <string>
#include <cstdint>
#include <Windows.h>
#include <algorithm>
#include "Data/NeuralDataFile.h"

enum LoadPathType : uint8_t
{
	InvalidLoadPath = 0,
	Single = 1,
	Multiple_Labels_Images = 2,
};

enum LoadProcessType : uint8_t 
{
	InvalidProcess = 0,
	ConcurrentProcess = 1,
	ParallelProcess = 2,
};

struct FileLoadConfig {
public:
	virtual const std::string FileDirectoryPath() = 0;
	virtual const std::vector<std::string> FileDirectoryPaths() = 0;
	virtual const NeuralDataObject_Type GetNeuralType()
	{
		return NeuralDataObject_Type::NrualDataObject_Invalid;
	}
	virtual LoadPathType GetLoadPathType() 
	{
		return LoadPathType::InvalidLoadPath;
	}
	virtual LoadProcessType GetLoadProcessType()
	{
		return LoadProcessType::InvalidProcess;
	}

	std::string GetCurrentRelativeDirectory() {
		char buffer[MAX_PATH];
		GetCurrentDirectory(MAX_PATH, buffer);
		return std::string(buffer);
	}

	std::string FixSlashes(const std::string& path) {
		std::string fixedPath = path;
		std::replace(fixedPath.begin(), fixedPath.end(), '\\', '/');
		return fixedPath;
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

			//std::string("D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/Data/Kaggle_Mnist/train-images.idx3-ubyte"),
			std::string(FixSlashes(GetCurrentRelativeDirectory()) + "/Data/Kaggle_Mnist/train-images.idx3-ubyte"),
			//std::string("D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/Data/Kaggle_Mnist/train-labels.idx1-ubyte")
			std::string(FixSlashes(GetCurrentRelativeDirectory()) + "/Data/Kaggle_Mnist/train-labels.idx1-ubyte")
		};
	}

	virtual LoadPathType GetLoadPathType() override 
	{
		return PathType;
	}

	virtual const NeuralDataObject_Type GetNeuralType() override 
	{
		return NeuralDataObject_Type::NeuralDataObject_MNIST_Digit;
	}

	virtual LoadProcessType GetLoadProcessType() override
	{
		return LoadProcessType::ParallelProcess;
	}
};