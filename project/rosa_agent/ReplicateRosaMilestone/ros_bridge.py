from message_bus import MessageBus

class ROSBridge:
    """Client-side proxy that forwards requests to the ROS adapter via NATS."""
    def __init__(self, bus: MessageBus):
        self.bus = bus

    async def list_nodes(self) -> str:
        return await self.bus.request("ros.nodes.list")

    async def list_topics(self) -> str:
        return await self.bus.request("ros.topics.list")

    async def get_topic_info(self, topic: str) -> str:
        return await self.bus.request("ros.topic.info", {"topic": topic})

    async def get_topic_publishers(self, topic: str) -> str:
        return await self.bus.request("ros.topic.publishers", {"topic": topic})

    async def get_topic_subscribers(self, topic: str) -> str:
        return await self.bus.request("ros.topic.subscribers", {"topic": topic})

    async def echo_topic(self, topic: str) -> str:
        return await self.bus.request("ros.topic.echo", {"topic": topic})
    
    async def subscribe_topic(self, topic: str) -> str:
        return await self.bus.request("ros.topic.subscribe", {"topic": topic}, timeout=5.0)
