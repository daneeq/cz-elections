# Capture System Configuration

## Environment Variables

The capture system now supports several configuration options to help with connection issues and rate limiting:

### HTTP Settings
- `CAPTURE_HTTP_TIMEOUT=30` - Request timeout in seconds (default: 30)
- `CAPTURE_MAX_RETRIES=3` - Maximum retry attempts for failed requests (default: 3)

### Rate Limiting
- `CAPTURE_STATEFUL_DELAY=2` - Delay between stateful feed requests in seconds (default: 2)
- `CAPTURE_BATCH_DELAY=1` - Delay between batch requests in seconds (default: 1)

### Capture Intervals
- `CAPTURE_STATEFUL_INTERVAL=60` - Stateful capture interval in seconds (default: 60)
- `CAPTURE_POLL_GRACE=5` - Grace period for polling cycles in seconds (default: 5)

### Data Storage
- `DATA_DIR=data` - Base data directory (default: data)
- `S3_BUCKET=` - S3 bucket for data storage (optional)
- `S3_PREFIX=` - S3 key prefix (optional)
- `AWS_REGION=eu-central-1` - AWS region for S3 (default: eu-central-1)

### Logging
- `LOG_LEVEL=INFO` - Logging level (DEBUG, INFO, WARNING, ERROR)

## Example .env file

```bash
# Copy this to .env and adjust as needed
LOG_LEVEL=INFO
CAPTURE_HTTP_TIMEOUT=45
CAPTURE_MAX_RETRIES=5
CAPTURE_STATEFUL_DELAY=3
CAPTURE_BATCH_DELAY=2
```

## Recent Improvements

The capture system has been enhanced with:

1. **Retry Logic**: Automatic retry with exponential backoff for connection failures
2. **Connection Pooling**: Reuse HTTP connections to reduce overhead
3. **Rate Limiting**: Configurable delays between requests to avoid overwhelming the server
4. **Better Error Handling**: Distinguishes between different types of connection failures
5. **User-Agent Header**: Proper browser-like User-Agent to avoid being blocked
6. **Configurable Timeouts**: Adjustable timeouts for different network conditions

## Troubleshooting Connection Issues

If you're still experiencing connection issues, try:

1. **Increase timeouts**: Set `CAPTURE_HTTP_TIMEOUT=60` or higher
2. **Increase retries**: Set `CAPTURE_MAX_RETRIES=5`
3. **Slow down requests**: Increase `CAPTURE_STATEFUL_DELAY=5` and `CAPTURE_BATCH_DELAY=3`
4. **Check network**: Verify the server can reach the target URLs
5. **Monitor logs**: Use `LOG_LEVEL=DEBUG` to see detailed connection attempts

## AWS Deployment

For AWS deployment, you may want to use these settings:

```bash
# More conservative settings for AWS
CAPTURE_HTTP_TIMEOUT=60
CAPTURE_MAX_RETRIES=5
CAPTURE_STATEFUL_DELAY=5
CAPTURE_BATCH_DELAY=3
LOG_LEVEL=INFO
```
