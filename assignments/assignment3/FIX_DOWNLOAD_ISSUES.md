# Fixing Download Issues - Network Timeouts

The log shows network timeout errors when trying to download from Google Cloud Storage. Here's how to fix them:

## Issues Identified

1. **Google Cloud Authentication Missing**
   - Error: "Could not locate the credentials file"
   - Solution: Authenticate with Google Cloud

2. **Network Timeouts**
   - Error: "transmission has been stuck... and will be aborted"
   - Solution: Add retry logic and better error handling

## Step 1: Authenticate with Google Cloud

```bash
# Authenticate with Google Cloud
gcloud auth application-default login

# This will open a browser window for authentication
# The dataset is publicly accessible but requires authentication
```

## Step 2: Verify Authentication

```bash
# Check if authentication worked
gcloud auth application-default print-access-token

# Should print a token, not an error
```

## Step 3: Run Download with Improved Error Handling

The updated `download_block_episodes.py` now includes:
- Retry logic for network timeouts (up to 3 attempts)
- Better error messages
- Progress indicators

```bash
cd /workspaces/eng-ai-agents/assignments/assignment3

python download_block_episodes.py \
    --episode-ids block_episode_ids.json \
    --max-episodes 5 \
    --output-dir data/block_episodes \
    --annotations droid_language_annotations.json
```

## Alternative: Use Local TFDS Cache

If network issues persist, you can try downloading the full dataset to a local cache first:

```bash
# Option 1: Let TFDS cache data locally
export TFDS_DATA_DIR=/path/to/local/cache

# Then run the download script
python download_block_episodes.py \
    --episode-ids block_episode_ids.json \
    --max-episodes 5 \
    --tfds-dir /path/to/local/cache
```

## Troubleshooting

### Still Getting Timeouts?

1. **Check Network Connection**
   ```bash
   ping storage.googleapis.com
   ```

2. **Try Again Later**
   - The dataset may be experiencing high load
   - Network timeouts can be temporary

3. **Download in Smaller Batches**
   ```bash
   # Download just 1 episode first to test
   python download_block_episodes.py \
       --episode-ids block_episode_ids.json \
       --max-episodes 1 \
       --output-dir data/block_episodes
   ```

4. **Use VPN or Different Network**
   - Some networks may have restrictions on large downloads
   - Try a different network connection

### Authentication Errors

If you see authentication errors:
```bash
# Re-authenticate
gcloud auth application-default login

# Check credentials
gcloud auth application-default print-access-token
```

### What the Script Does Now

1. **Retry Logic**: Automatically retries up to 3 times on network timeouts
2. **Better Error Messages**: Shows exactly what went wrong
3. **Progress Tracking**: Shows scan progress every 1000 episodes
4. **Graceful Failure**: Skips problematic episodes and continues

## Expected Behavior

After fixing:
- ✅ Authentication: Should authenticate successfully
- ✅ Dataset Loading: Should connect to Google Cloud Storage
- ✅ Episode Scanning: Should scan through episodes
- ✅ Downloading: Should download matched episodes with retry on timeouts

If you still experience issues after authentication, the network timeouts may be temporary. The script will automatically retry and skip problematic episodes.


