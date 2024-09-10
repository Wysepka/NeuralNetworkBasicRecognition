# CMakeList.txt: plik projektu CMake najwyższego poziomu, wykonaj konfigurację globalną
# i uwzględnij podprojekty w tym miejscu.
#

# Włącz przeładowywanie kodu XAML na gorąco dla kompilatorów MSVC, jeśli jest to obsługiwane.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

cmake_minimum_required(VERSION 3.10)

# Set the project name
project(NeuralNetwork_DigitRecognition)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add subdirectories
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(lib)

# Add the executable
add_executable(NeuralNetwork_DigitRecognition src/private/main/main.cpp "include/Engine/Engine.h" "include/Config/FileLoadConfig.h" "include/Data/NeuralDataFile.h" "include/Loader/FileLoader.h" "include/Settings/FileLoadSettings.h" "include/Config/ProjectConfig.h" "include/Log/Logger.h" "include/Rendering/RenderingSystem.h" "include/Rendering/TextureLoader.h")

find_package(OpenGL REQUIRED)

# Link the sources and libraries to the main executable
target_link_libraries(NeuralNetwork_DigitRecognition PRIVATE NeuralNetwork_Sources NeuralNetwork_Headers ${LIBS} MNISTReader glfw glad glm imgui ${LIBRARIES} OpenGL::GL)