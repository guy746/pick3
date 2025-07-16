import redis
import json
import logging
import time
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Base class for all agents providing common functionality"""
    
    def __init__(self, name, subscribe_channels=None, publish_channel=None):
        self.name = name
        self.subscribe_channels = subscribe_channels or []
        self.publish_channel = publish_channel
        # Redis connection - using host.docker.internal for container-to-host communication
        # Note: localhost won't work from inside Docker containers
        self.redis = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.pubsub = self.redis.pubsub()
        self.logger = self.setup_logger()
        self.metrics = {
            'messages_received': 0,
            'messages_processed': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
    def setup_logger(self):
        """Configure structured logging"""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "agent": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    def subscribe(self):
        """Subscribe to Redis channels"""
        if self.subscribe_channels:
            self.pubsub.subscribe(self.subscribe_channels)
            self.logger.info(f"Subscribed to channels: {', '.join(self.subscribe_channels)}")
            
    def publish(self, event_type, data):
        """Publish event to Redis"""
        if not self.publish_channel:
            self.logger.warning("No publish channel configured")
            return
            
        event = {
            'event': event_type,
            'timestamp': time.time(),
            'agent': self.name,
            'data': data
        }
        self.redis.publish(self.publish_channel, json.dumps(event))
        self.logger.debug(f"Published {event_type} event")
        
    def record_metric(self, name, value=1):
        """Record performance metric"""
        self.metrics[name] = self.metrics.get(name, 0) + value
        
    def get_metrics(self):
        """Return current metrics snapshot"""
        uptime = time.time() - self.metrics['start_time']
        return {
            **self.metrics,
            'uptime': uptime,
            'msg_rate': self.metrics['messages_received'] / uptime if uptime > 0 else 0
        }
        
    @abstractmethod
    def handle_message(self, channel, message):
        """Abstract method to handle incoming messages (must be implemented by subclasses)"""
        pass
        
    def run(self):
        """Main agent loop"""
        self.subscribe()
        self.logger.info(f"{self.name} agent started")
        
        for message in self.pubsub.listen():
            if message['type'] != 'message':
                continue
                
            try:
                self.record_metric('messages_received')
                data = json.loads(message['data'])
                self.handle_message(message['channel'], data)
                self.record_metric('messages_processed')
            except Exception as e:
                self.logger.error(f"Error processing message: {e}", exc_info=True)
                self.record_metric('errors')
