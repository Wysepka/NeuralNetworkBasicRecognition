# Neural Network Basic Recognition

A C++ implementation of a neural network for pattern recognition, specifically designed for digit recognition using the MNIST dataset. This project demonstrates the fundamentals of neural network architecture, backpropagation, and machine learning concepts.

## ⚠️ Project Status

**This project is currently not functional and has been discontinued.** The developer lost faith in debugging the application, though valuable learning was gained about C++ and neural networks during development.

## 🏗️ Project Overview

This neural network implementation features:

- **Multi-layer perceptron** architecture
- **Backpropagation** algorithm for training
- **Parallel batch processing** capabilities
- **OpenGL-based visualization** using ImGui
- **MNIST dataset** integration for digit recognition
- **Modular design** with separate components for different functionalities

## 📁 Project Structure

```
wysepka-neuralnetworkbasicrecognition/
├── README.md
├── CMakeLists.txt
├── CMakePresets.json
├── Data/
│   └── Kaggle_Mnist/
│       ├── t10k-images.idx3-ubyte
│       ├── t10k-labels.idx1-ubyte
│       ├── train-images.idx3-ubyte
│       └── train-labels.idx1-ubyte
├── include/
│   ├── Config/           # Configuration management
│   ├── Core/             # Core enums and utilities
│   ├── Data/             # Data handling and batching
│   ├── Engine/           # Main engine coordination
│   ├── Event/            # Message bus system
│   ├── Loader/           # File loading utilities
│   ├── Log/              # Logging system
│   ├── Neural/           # Neural network core components
│   ├── Rendering/        # OpenGL rendering system
│   └── Settings/         # Application settings
├── lib/                  # External libraries
│   ├── glad/            # OpenGL loader
│   ├── glfw/            # Window management
│   ├── glm/             # Mathematics library
│   ├── imgui/           # Immediate mode GUI
│   └── MNISTReader/     # MNIST data reader
├── NeuralNetwork_DigitRecognition/
└── src/                 # Implementation files
    └── private/
        ├── Config/
        ├── Data/
        ├── Engine/
        ├── Event/
        ├── Loader/
        ├── Log/
        ├── main/
        ├── Neural/
        ├── Rendering/
        └── Settings/
```

## 🧠 Neural Network Architecture

### Core Components

1. **Layer System**
   - Input, Hidden, and Output layers
   - Configurable layer sizes
   - Weight and bias management
   - Gradient calculation and application

2. **Activation Functions**
   - Sigmoid
   - TanH (Hyperbolic Tangent)
   - ReLU (Rectified Linear Unit)
   - SiLU (Sigmoid Linear Unit)
   - Softmax

3. **Cost Functions**
   - Mean Squared Error (MSE)
   - Cross Entropy

4. **Training Features**
   - Batch processing
   - Parallel computation support
   - Learning rate decay
   - Momentum optimization
   - Weight decay

### Network Configuration

The neural network supports:
- Configurable input layer size (784 for MNIST)
- Multiple hidden layers with custom node counts
- Output layer with 10 nodes (digits 0-9)
- Batch size configuration
- Epoch count settings
- Parallel processing options

## 🎯 Features

### Data Processing
- **MNIST Dataset Support**: Built-in support for MNIST digit recognition
- **Data Batching**: Efficient batch processing for large datasets
- **Data Normalization**: Automatic pixel value normalization
- **File Loading**: Robust file loading system with error handling

### Training System
- **Backpropagation**: Full implementation of backpropagation algorithm
- **Gradient Descent**: Optimized gradient descent with momentum
- **Learning Rate Scheduling**: Dynamic learning rate adjustment
- **Parallel Processing**: Multi-threaded batch processing
- **Progress Tracking**: Real-time training progress monitoring

### Visualization
- **OpenGL Rendering**: Hardware-accelerated graphics
- **ImGui Interface**: Modern, immediate mode GUI
- **Texture Display**: Real-time display of MNIST images
- **Training Visualization**: Live training progress and statistics

### System Architecture
- **Message Bus**: Event-driven architecture for component communication
- **Modular Design**: Clean separation of concerns
- **Configuration Management**: Flexible configuration system
- **Logging System**: Comprehensive logging capabilities

## 🛠️ Technical Stack

### Core Technologies
- **C++17**: Modern C++ features and standards
- **CMake**: Cross-platform build system
- **OpenGL 4.6**: Graphics rendering
- **GLFW**: Window management and input handling

### External Libraries
- **GLAD**: OpenGL function loading
- **GLM**: Mathematics library
- **Dear ImGui**: Immediate mode GUI
- **MNIST Reader**: Dataset loading utilities

### Build System
- **CMake 3.10+**: Required minimum version
- **Ninja**: Build generator (optional)
- **MSVC/Clang**: Supported compilers

## 🚀 Building the Project

### Prerequisites
- C++17 compatible compiler
- CMake 3.10 or higher
- OpenGL 4.6 compatible graphics driver

### Build Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd wysepka-neuralnetworkbasicrecognition
   ```

2. **Configure with CMake**
   ```bash
   mkdir build
   cd build
   cmake ..
   ```

3. **Build the project**
   ```bash
   cmake --build .
   ```

### Build Presets

The project includes several CMake presets:
- `x64-debug`: 64-bit debug build
- `x64-release`: 64-bit release build
- `x86-debug`: 32-bit debug build
- `x86-release`: 32-bit release build

## 📊 Usage

### Running the Application

1. **Ensure MNIST data is present** in the `Data/Kaggle_Mnist/` directory
2. **Run the executable**:
   ```bash
   ./NeuralNetwork_DigitRecognition
   ```

### Configuration

The neural network can be configured through the `NeuralNetworkConfig` structure:

```cpp
// Example configuration
NeuralNetworkConfig config(
    784,                    // Input layer size
    hiddenLayers,          // Hidden layer configuration
    10,                    // Output layer size
    activationFunction,    // Activation function
    costFunction,          // Cost function
    batchSize,            // Batch size
    epochs,               // Number of epochs
    useParallel,          // Enable parallel processing
    maxParallelThreads,   // Maximum parallel threads
    logConfig            // Logging configuration
);
```

## 🔧 Key Components

### Neural Network Core (`include/Neural/`)

- **`NeuralNetwork.h/cpp`**: Main network implementation
- **`Layer.h/cpp`**: Individual layer management
- **`LayerBuffer.h`**: Layer data buffers
- **`Activation.h`**: Activation function implementations
- **`Cost.h`**: Cost function implementations

### Data Management (`include/Data/`)

- **`NeuralDataFile.h/cpp`**: Data file handling
- **`NeuralDataBatch.h/cpp`**: Batch processing utilities

### Rendering System (`include/Rendering/`)

- **`RenderingSystem.h/cpp`**: OpenGL rendering engine
- **`TextureLoader.h/cpp`**: Texture loading and management
- **`UIRenderer.h/cpp`**: User interface rendering

### Event System (`include/Event/`)

- **`MessageBus.h/cpp`**: Event-driven communication
- **`Messages.h`**: Message definitions

## 📈 Performance Considerations

### Optimization Features
- **Parallel Batch Processing**: Multi-threaded training
- **Memory Management**: Efficient memory allocation
- **GPU Acceleration**: OpenGL-based rendering
- **Batch Normalization**: Improved training stability

### Known Limitations
- **Single-threaded Forward/Backward Pass**: Individual training iterations are sequential
- **Memory Usage**: Large datasets may require significant RAM
- **Training Time**: Complex networks may require extended training periods

## 🐛 Known Issues

As mentioned in the project status, this implementation has several issues:

1. **Non-functional Training**: The neural network training process is not working correctly
2. **Debugging Challenges**: The developer encountered difficulties in debugging the application
3. **Architecture Complexity**: The modular design may have introduced integration issues

## 📚 Learning Outcomes

Despite the project's non-functional status, valuable insights were gained:

- **C++ Modern Features**: Extensive use of C++17 features
- **Neural Network Theory**: Deep understanding of backpropagation and gradient descent
- **Software Architecture**: Event-driven design patterns
- **Graphics Programming**: OpenGL and ImGui integration
- **Build Systems**: CMake configuration and cross-platform development

## 🤝 Contributing

While this project is discontinued, the codebase serves as a valuable learning resource for:

- Neural network implementation concepts
- C++ software architecture patterns
- Graphics programming with OpenGL
- Event-driven system design

## 📄 License

This project is provided as-is for educational purposes. Please refer to the original repository for licensing information.

## 🙏 Acknowledgments

- **MNIST Dataset**: Standard benchmark for digit recognition
- **OpenGL Community**: Graphics programming resources
- **Dear ImGui**: Excellent immediate mode GUI library
- **GLFW**: Cross-platform window management

---

*This README documents the project structure and implementation details for educational purposes. The project itself is not functional but provides valuable insights into neural network implementation and C++ software development.* 
