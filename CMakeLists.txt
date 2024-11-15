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
#add_executable(NeuralNetwork_DigitRecognition src/private/main/main.cpp "include/Engine/Engine.h" "include/Config/FileLoadConfig.h" "include/Data/NeuralDataFile.h" "include/Loader/FileLoader.h" "include/Settings/FileLoadSettings.h" "include/Config/ProjectConfig.h" "include/Log/Logger.h" "include/Rendering/RenderingSystem.h" "include/Rendering/TextureLoader.h" "include/Event/Messages.h" "include/Event/MessageBus.h" "include/Neural/NeuralNetwork.h" "include/Neural/Layer.h" "include/Neural/LayerBuffer.h" "include/Neural/IActivation.h" "include/Neural/Activation.h" "include/Neural/ICost.h" "include/Neural/Cost.h" "include/Config/NeuralNetworkConfig.h")
add_executable(NeuralNetwork_DigitRecognition src/private/main/main.cpp)

find_package(OpenGL REQUIRED)

# Link the sources and libraries to the main executable
target_link_libraries(NeuralNetwork_DigitRecognition PRIVATE NeuralNetwork_Sources NeuralNetwork_Headers ${LIBS} MNISTReader glfw glad glm imgui ${LIBRARIES} OpenGL::GL)

# Define the source folder to be copied
set(SOURCE_DATA_FOLDER ${CMAKE_SOURCE_DIR}/Data)

# Define the destination folder (in the build directory)
set(DESTINATION_DATA_FOLDER ${CMAKE_BINARY_DIR}/Data)

# Copy the folder recursively during configuration
file(COPY ${SOURCE_DATA_FOLDER} DESTINATION ${DESTINATION_DATA_FOLDER})

# Add custom commands to copy the folder whenever you build
add_custom_command(
    TARGET NeuralNetwork_DigitRecognition
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${SOURCE_DATA_FOLDER} ${DESTINATION_DATA_FOLDER}
)