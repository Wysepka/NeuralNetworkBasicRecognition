# Include directories (necessary headers)
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add all .cpp files in the src directory
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/**/*.cpp")

# Create a library from the source files
add_library(NeuralNetwork_Sources ${SRC_FILES} "private/Initializer/Initializer.cpp" "private/Settings/FileLoadSettings.cpp" "private/Loader/FileLoader.cpp" "private/Log/Logger.cpp" "private/Config/ProjectConfig.cpp")

# (Optional) Link any necessary libraries if specific to this module
target_link_libraries(NeuralNetwork_Sources PRIVATE ${LIBS})