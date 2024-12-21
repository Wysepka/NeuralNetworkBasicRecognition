# Include the necessary directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Optionally, organize and manage headers
file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/include/**/*.h")

# Create an interface library for headers (if they are header-only)
add_library(NeuralNetwork_Headers INTERFACE
        Core/EnumsContainer.h
        Neural/NeuralNetworkUtility.h
        Neural/NeuralNetworkResult.h
        Neural/NeuralNetworkController.h
        Neural/Cost.h
        Data/NeuralDataBatch.h
)

# Specify include directories for this interface library
target_include_directories(NeuralNetwork_Headers INTERFACE ${CMAKE_SOURCE_DIR}/include)

# Optionally, you can install the headers if this is meant to be a distributable library
install(FILES ${HEADER_FILES} DESTINATION include)