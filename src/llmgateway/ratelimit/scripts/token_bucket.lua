-- Token bucket implementation
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local requested = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

-- Get current state
local state = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(state[1]) or capacity
local last_refill = tonumber(state[2]) or now

-- Calculate refill
local elapsed = math.max(0, now - last_refill)
local refill = elapsed * rate
tokens = math.min(capacity, tokens + refill)

-- Check if enough tokens
if tokens >= requested then
  tokens = tokens - requested
  redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
  redis.call('EXPIRE', key, 3600)
  return {1, tokens, 0}  -- allowed, remaining, retry_after
else
  local needed = requested - tokens
  local retry_after = needed / rate
  return {0, tokens, retry_after}  -- denied, remaining, retry_after
end
