import asyncio, json, os
from typing import Any, Dict, Protocol, List
from nats.aio.client import Client as NATS

class MessageBus:
    """Concise NATS wrapper."""
    def __init__(self, url: str = "nats://localhost:4222"):
        self.url = url
        self.nc = NATS()

    async def connect(self):
        await self.nc.connect(self.url)
        print(f"Connected to NATS: {self.url}")

    async def disconnect(self):
        await self.nc.close()

    async def request(self, subject: str, payload: Dict = None, timeout: float = 5.0) -> Any:
        resp = await self.nc.request(subject, json.dumps(payload or {}).encode(), timeout=timeout)
        return json.loads(resp.data.decode()).get("data")

    async def subscribe(self, subject: str, callback):
        async def handler(msg):
            data = json.loads(msg.data.decode())
            res = await callback(data)
            if msg.reply: await self.nc.publish(msg.reply, json.dumps({"data": res}).encode())
        await self.nc.subscribe(subject, cb=handler)

class Agent(Protocol):
    """Minimal Agent Protocol."""
    def process(self, user_input: str) -> str: ...