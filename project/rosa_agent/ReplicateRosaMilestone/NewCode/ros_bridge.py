from core import MessageBus

class ROSBridge:
    """Client-side proxy that forwards requests to the ROS adapter via NATS."""
    def __init__(self, bus: MessageBus):
        self.bus = bus

    # CRITICAL FIX: Added **kwargs to ALL methods below.
    # This allows the function to silently ignore "Action Input: None" 
    # which the LLM frequently sends by mistake.

    async def list_nodes(self) -> str:
        return await self.bus.request("ros.nodes.list")

    async def list_topics(self) -> str:
        return await self.bus.request("ros.topics.list")

    async def get_topic_info(self, topic: str, **kwargs) -> str:
        return await self.bus.request("ros.topic.info", {"topic": topic})

    async def echo_topic(self, topic: str, **kwargs) -> str:
        return await self.bus.request("ros.topic.echo", {"topic": topic})
    
    async def subscribe_topic(self, topic: str, **kwargs) -> str:
        # Listen to topic once and return the message
        return await self.bus.request(
            "ros.topic.subscribe", 
            {"topic": topic}, 
            timeout=5.0
        )