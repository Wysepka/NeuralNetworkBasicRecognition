#include "Event/MessageBus.h"

std::unordered_map<std::type_index, std::vector<std::function<void(const std::shared_ptr<Message>&)>>> MessageBus::subscribers;

template void MessageBus::Subscribe<EndLoadingFile>(const std::function<void(const std::shared_ptr<EndLoadingFile>&)>& callback);
template void MessageBus::Publish<EndLoadingFile>(const std::shared_ptr<EndLoadingFile>& message);

template void MessageBus::Subscribe<NeuralNetworkInitialized>(const std::function<void(const std::shared_ptr<NeuralNetworkInitialized>&)>& callback);
template void MessageBus::Publish<NeuralNetworkInitialized>(const std::shared_ptr<NeuralNetworkInitialized>& message);

template<typename T>
void MessageBus::ValidateMessageType() {
	static_assert(std::is_base_of<Message, T>::value, "T must be derived from Message");
}

template<typename T>
void MessageBus::Subscribe(const std::function<void(const std::shared_ptr<T>&)>& callback) {
	ValidateMessageType<T>();  // Ensure T is derived from Message

	std::type_index type = std::type_index(typeid(T));

	// Create a wrapper to cast to the correct type before calling the subscriber's callback
	subscribers[type].push_back([callback](const std::shared_ptr<Message>& message) {
		callback(std::static_pointer_cast<T>(message));
		});
}

// Static method to publish a message
template<typename T>
void MessageBus::Publish(const std::shared_ptr<T>& message) {
	ValidateMessageType<T>();  // Ensure T is derived from Message

	std::type_index type = std::type_index(typeid(T));

	if (subscribers.find(type) != subscribers.end()) {
		for (const auto& callback : subscribers[type]) {
			callback(message); // Call each subscriber with the message
		}
	}
}