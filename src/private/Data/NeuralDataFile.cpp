
#include "../include/Data/NeuralDataFile.h"

NeuralDataObject_Type NeuralDataObject::GetNeuralDataObject_Type()
{
    return neuralDataObjectType;
}

void NeuralDataObject::SetFlatObjectsPixelsArray(std::shared_ptr<std::vector<uint8_t>> flatObjectPixelsArray) 
{
	this->flatObjectPixelsArray = flatObjectPixelsArray;
}

void NeuralDataObject::SetLabel(int label)
{
	this->label = label;
}

void NeuralDataObject::SetDimensions(int xDim, int yDim)
{
	this->dimensionX = xDim;
	this->dimensionY = yDim;
}

int NeuralDataObject::GetLabel()
{
	return label;
}

int NeuralDataObject::GetXDim()
{
	return dimensionX;
}

int NeuralDataObject::GetYDim()
{
	return dimensionY;
}

std::vector<uint8_t> NeuralDataObject::GetFlatObjectPixelsArray()
{
	return *flatObjectPixelsArray;
}


//==========================================================================================================
//==============================||| Below is NeuralDataFile Members |||=====================================
//==========================================================================================================

std::shared_ptr<NeuralDataFile> NeuralDataFile::CreateInstanceOfNeuralDataFile(NeuralDataObject_Type type)
{
	std::shared_ptr<NeuralDataFile> dataFile;
	switch (type)
	{
	case Invalid:
		dataFile = std::make_shared<NeuralDataFile>();
		dataFile->SetAsInvalid();
		return dataFile;
	case MNIST_Digit:
		return std::make_shared<NeuralDataFile_MNIST_Digits>();
	default:
		return std::make_shared<NeuralDataFile>();
	}
}

void NeuralDataFile::SetAsInvalid() {
	invalid = true;
}

void NeuralDataFile::Initialize(int elementsCount) 
{
	this->elementsCount = elementsCount;
	this->neuralDataObjects = std::make_shared<std::vector<std::shared_ptr<NeuralDataObject>>>();
}

void NeuralDataFile::AddDataObject(std::shared_ptr<NeuralDataObject> dataObject) 
{
	this->neuralDataObjects->push_back(dataObject);
}

const std::vector<std::shared_ptr<NeuralDataObject>>& NeuralDataFile::GetNeuralDataObjects() const
{
	return *neuralDataObjects;
}
