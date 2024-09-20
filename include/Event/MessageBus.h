#pragma once
#include <unordered_map>
#include <typeindex>
#include <functional>
#include <memory>
#include "Messages.h"

class MessageBus
{
private:
	// Map of type_index to vector of callbacks (subscribers) for each message type
	static std::unordered_map<std::type_index, std::vector<std::function<void(const std::shared_ptr<Message>&)>>> subscribers;

	// Helper function to ensure T is derived from Message
	template<typename T>
	static void ValidateMessageType();

public:
	// Static method to subscribe to a specific type of message
	template<typename T>
	static void Subscribe(const std::function<void(const std::shared_ptr<T>&)>& callback);

	// Static method to publish a message
	template<typename T>
	static void Publish(const std::shared_ptr<T>& message);
};