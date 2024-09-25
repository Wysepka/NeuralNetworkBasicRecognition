# Install script for directory: D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/include

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/out/install/x64-debug")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/include/Config/FileLoadConfig.h"
    "D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/include/Config/NeuralNetworkConfig.h"
    "D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/include/Config/ProjectConfig.h"
    "D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/include/Data/NeuralDataFile.h"
    "D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/include/Engine/Engine.h"
    "D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/include/Event/MessageBus.h"
    "D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/include/Event/Messages.h"
    "D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/include/Loader/FileLoader.h"
    "D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/include/Log/Logger.h"
    "D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/include/Neural/Activation.h"
    "D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/include/Neural/Cost.h"
    "D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/include/Neural/IActivation.h"
    "D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/include/Neural/ICost.h"
    "D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/include/Neural/Layer.h"
    "D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/include/Neural/LayerBuffer.h"
    "D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/include/Neural/NeuralNetwork.h"
    "D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/include/Neural/NeuralTrainer.h"
    "D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/include/Rendering/RenderingSystem.h"
    "D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/include/Rendering/TextureLoader.h"
    "D:/Projekty/NeuralNetwork/NeuralNetwork_DigitRecognition/include/Settings/FileLoadSettings.h"
    )
endif()

