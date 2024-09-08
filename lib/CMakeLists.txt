# Add the libraries (assuming these are pre-built libraries)

#add_library(CustomLibrary STATIC IMPORTED)

# Specify the location of the library files

#set_target_properties(CustomLibrary PROPERTIES
#    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/lib/CustomLibrary.a"
#)

# If you have multiple libraries, add them similarly and include them in the LIBS variable

#set(LIBS CustomLibrary)

# If needed, link libraries to other targets

#target_link_libraries(NeuralNetwork_Sources PRIVATE ${LIBS})

# CMakeLists.txt in the lib directory

# Specify the minimum version of CMake
cmake_minimum_required(VERSION 3.10)

# Optionally, you can still declare an interface library if needed
add_library(MNISTReader INTERFACE)

# Specify the directory where the .hpp files are located
target_include_directories(MNISTReader INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/mnist   # Assuming .hpp files are in the include folder
)