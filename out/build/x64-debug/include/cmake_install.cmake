# Install script for directory: C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/include

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/out/install/x64-debug")
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
    "C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/include/Config/FileLoadConfig.h"
    "C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/include/Config/NeuralNetworkConfig.h"
    "C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/include/Config/ProjectConfig.h"
    "C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/include/Data/NeuralDataFile.h"
    "C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/include/Engine/Engine.h"
    "C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/include/Event/MessageBus.h"
    "C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/include/Event/Messages.h"
    "C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/include/Loader/FileLoader.h"
    "C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/include/Log/Logger.h"
    "C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/include/Neural/Activation.h"
    "C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/include/Neural/Cost.h"
    "C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/include/Neural/IActivation.h"
    "C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/include/Neural/ICost.h"
    "C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/include/Neural/Layer.h"
    "C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/include/Neural/LayerBuffer.h"
    "C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/include/Neural/NeuralNetwork.h"
    "C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/include/Neural/NeuralTrainer.h"
    "C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/include/Rendering/RenderingSystem.h"
    "C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/include/Rendering/TextureLoader.h"
    "C:/Dev/Projekty/NeuralNetworkDigitRecognition/NeuralNetworkBasicRecognition/include/Settings/FileLoadSettings.h"
    )
endif()

