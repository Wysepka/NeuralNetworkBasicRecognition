# CMakeList.txt: projekt CMake dla elementu NeuralNetwork_DigitRecognition, dołącz źródło i zdefiniuj
# tutaj logikę specyficzną dla projektu.
#

# Dodaj źródło do pliku wykonywalnego tego projektu.
add_executable (NeuralNetwork_DigitRecognition "NeuralNetwork_DigitRecognition.cpp" "NeuralNetwork_DigitRecognition.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET NeuralNetwork_DigitRecognition PROPERTY CXX_STANDARD 20)
endif()

# TODO: Dodaj testy i zainstaluj elementy docelowe w razie potrzeby.
