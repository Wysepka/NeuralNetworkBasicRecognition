#include "Config/ProjectConfig.h"

std::array<std::shared_ptr<FileLoadConfig>, 1> ProjectConfig::LoadTypesQueue = { std::make_shared<FileLoadConfig_MNIST>() };