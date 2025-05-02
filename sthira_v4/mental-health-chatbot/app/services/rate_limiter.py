from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
import redis

class DistributedRateLimiter:
    def __init__(self, redis_url: str = 'redis://localhost:6379'):
        """
        Initialize distributed rate limiter with Redis.
        
        :param redis_url: Redis connection URL
        """
        self.redis_client = redis.from_url(redis_url)
        
        self.limiter = Limiter(
            key_func=get_remote_address,
            storage_uri=redis_url,
            default_limits=[
                "10 per minute",  # Basic rate limit
                "100 per day"     # Daily limit
            ]
        )
    
    def limit_request(self, request: Request):
        """
        Apply rate limiting to incoming requests.
        
        :param request: FastAPI request object
        """
        client_ip = get_remote_address(request)
        
        # Track and limit requests
        request_key = f"rate_limit:{client_ip}"
        current_count = self.redis_client.incr(request_key)
        
        if current_count > 10:  # Per minute limit
            raise RateLimitExceeded("Too many requests")
        
        # Set expiration for rate limit tracking
        self.redis_client.expire(request_key, 60)  # 1-minute window