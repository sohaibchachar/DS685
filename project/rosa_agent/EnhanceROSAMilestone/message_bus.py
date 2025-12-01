import json
import asyncio
from typing import Optional, Callable, Dict, Any
from abc import ABC, abstractmethod

try:
    import nats
    from nats.aio.client import Client as NATS
    from nats.js import api
    NATS_AVAILABLE = True
except ImportError:
    NATS_AVAILABLE = False
    print("Warning: nats-py not installed. Install with: pip install nats-py")


class MessageBus(ABC):
    
    @abstractmethod
    async def publish(self, subject: str, message: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    async def subscribe(self, subject: str, callback: Callable) -> None:
        pass
    
    @abstractmethod
    async def request(self, subject: str, message: Dict[str, Any], timeout: float = 5.0) -> Dict[str, Any]:
        pass


class NATSMessageBus(MessageBus):
    
    def __init__(self, nats_url: str = "nats://localhost:4222"):
        self.nats_url = nats_url
        self.nc: Optional[NATS] = None
        self.js: Optional[api.JetStreamContext] = None
        self._connected = False
    
    async def connect(self):
        if not NATS_AVAILABLE:
            raise RuntimeError("nats-py is not installed. Install with: pip install nats-py")
        
        self.nc = await nats.connect(self.nats_url)
        self.js = self.nc.jetstream()
        self._connected = True
        print(f"Connected to NATS at {self.nats_url}")
    
    async def disconnect(self):
        if self.nc:
            await self.nc.close()
            self._connected = False
    
    async def publish(self, subject: str, message: Dict[str, Any]) -> None:
        if not self._connected:
            raise RuntimeError("Not connected to NATS. Call connect() first.")
        
        payload = json.dumps(message).encode()
        await self.nc.publish(subject, payload)
    
    async def subscribe(self, subject: str, callback: Callable) -> None:
        if not self._connected:
            raise RuntimeError("Not connected to NATS. Call connect() first.")
        
        async def message_handler(msg):
            try:
                data = json.loads(msg.data.decode())
                res = await callback(data)
                if msg.reply:
                    await self.nc.publish(msg.reply, json.dumps({"data": res}).encode())
            except Exception as e:
                print(f"Error handling message: {e}")
        
        await self.nc.subscribe(subject, cb=message_handler)
    
    async def request(self, subject: str, message: Dict[str, Any] = None, timeout: float = 5.0) -> Any:
        if not self._connected:
            raise RuntimeError("Not connected to NATS. Call connect() first.")
        
        payload = json.dumps(message or {}).encode()
        response = await self.nc.request(subject, payload, timeout=timeout)
        return json.loads(response.data.decode()).get("data")


class MockMessageBus(MessageBus):
    
    def __init__(self):
        self.messages: Dict[str, list] = {}
        self.subscribers: Dict[str, list] = {}
    
    async def publish(self, subject: str, message: Dict[str, Any]) -> None:
        if subject not in self.messages:
            self.messages[subject] = []
        self.messages[subject].append(message)

        if subject in self.subscribers:
            for callback in self.subscribers[subject]:
                await callback(message)
    
    async def subscribe(self, subject: str, callback: Callable) -> None:
        if subject not in self.subscribers:
            self.subscribers[subject] = []
        self.subscribers[subject].append(callback)
    
    async def request(self, subject: str, message: Dict[str, Any], timeout: float = 5.0) -> Dict[str, Any]:
        return {"status": "ok", "data": None}

