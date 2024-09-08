

# Specify the minimum version of CMake
cmake_minimum_required(VERSION 3.10)

# Optionally, you can still declare an interface library if needed

set(GLAD_SOURCES 
    glad/src/glad.c
)

add_library(imgui STATIC     
	imgui/backends/imgui_impl_glfw.cpp
    imgui/backends/imgui_impl_opengl3.cpp
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp)

add_library(glad STATIC ${GLAD_SOURCES})
add_library(MNISTReader INTERFACE)

add_subdirectory(glfw)
add_subdirectory(glm)

include_directories(glm)
include_directories(${CMAKE_SOURCE_DIR}/lib/glfw/include)  # Adds the GLFW include path
include_directories(${CMAKE_SOURCE_DIR}/lib/imgui)

set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)

set(LIBRARIES glfw glad glm imgui ${LIBRARIES} PARENT_SCOPE)

target_include_directories(glad PUBLIC glad/include)
target_include_directories(imgui PUBLIC 
    ${CMAKE_SOURCE_DIR}/lib/imgui
    ${CMAKE_SOURCE_DIR}/lib/imgui/backends)

# Specify the directory where the .hpp files are located
target_include_directories(MNISTReader INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/mnist   # Assuming .hpp files are in the include folder
)
