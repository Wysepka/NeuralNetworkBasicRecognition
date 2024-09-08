#pragma once
#include <vector>
#include <memory>
#include <cstdint>

enum NeuralDataObject_Type : uint8_t
{
	Invalid = 0,
	MNIST_Digit = 1,
};

class NeuralDataObject 
{
private:
	int dimensionX;
	int dimensionY;
	int label;
	std::shared_ptr<std::vector<uint8_t>> flatObjectPixelsArray;
	NeuralDataObject_Type neuralDataObjectType;

public:
	NeuralDataObject_Type GetNeuralDataObject_Type();
	void SetFlatObjectsPixelsArray(std::shared_ptr<std::vector<uint8_t>> flatObjectPixelsArray);
	void SetLabel(int label);
	void SetDimensions(int xDim, int yDim);

	//Getters
	int GetLabel();
	int GetXDim();
	int GetYDim();
	std::vector<uint8_t> GetFlatObjectPixelsArray();
};

class NeuralDataObject_MNIST_Digit : NeuralDataObject
{

};

class NeuralDataFile 
{
private:
	bool invalid;
	int elementsCount;
	std::shared_ptr<std::vector<std::shared_ptr<NeuralDataObject>>> neuralDataObjects;

public:
	virtual ~NeuralDataFile() = default;  // virtual destructor
	static std::shared_ptr<NeuralDataFile> CreateInstanceOfNeuralDataFile(NeuralDataObject_Type type);
	void SetAsInvalid();

	void Initialize(int elementsCount);
	void AddDataObject(std::shared_ptr<NeuralDataObject> dataObject);

	//Getters
	const std::vector<std::shared_ptr<NeuralDataObject>>& GetNeuralDataObjects() const;
};

class NeuralDataFile_MNIST_Digits : public NeuralDataFile
{
public:
	static const int X_DIMENSION = 28;
	static const int Y_DIMENSION = 28;
};