# Manual Download Guide for DROID Episodes

This guide shows how to download DROID episodes manually when automatic streaming has network issues.

## Method 1: Using gsutil (Recommended)

Google Cloud Storage utility (`gsutil`) can download files directly and is more reliable than streaming.

### Install gsutil

```bash
# If you have gcloud installed, gsutil comes with it
gcloud components install gsutil

# Or install standalone
# See: https://cloud.google.com/storage/docs/gsutil_install
```

### Download Episodes by ID

Unfortunately, DROID doesn't provide direct file paths by episode ID. The episodes are stored in TFRecord files that need to be processed.

### Alternative: Download TFRecord Files First

```bash
# Create directory for TFRecord files
mkdir -p data/droid_tfrecords

# Download specific TFRecord files (if you know which ones contain your episodes)
# The dataset has 2048 TFRecord files (droid_101-train.tfrecord-00000-of-02048, etc.)
gsutil -m cp -r gs://gresearch/robotics/droid/1.0.1/droid_101-train.tfrecord-00000-of-02048 data/droid_tfrecords/

# Download multiple files
gsutil -m cp -r gs://gresearch/robotics/droid/1.0.1/droid_101-train.tfrecord-0000*.tfrecord data/droid_tfrecords/

# Note: Each TFRecord file is ~64MB, so downloading many files will take time
```

### Then Process with TFDS

After downloading TFRecord files locally:

```bash
python download_block_episodes.py \
    --episode-ids block_episode_ids.json \
    --max-episodes 5 \
    --tfds-dir /path/to/local/tfds/data \
    --output-dir data/block_episodes \
    --annotations droid_language_annotations.json
```

## Method 2: Use TFDS with Full Download

Download the entire dataset to local cache, then process:

```python
import tensorflow_datasets as tfds

# Download full dataset to local cache
# This will take a LONG time and LOTS of space (~1.7TB)
ds = tfds.load(
    "droid",
    data_dir="gs://gresearch/robotics",
    split="train",
    download=True,  # Download full dataset
)

# Episodes will be cached locally at:
# ~/.cache/tensorflow_datasets/droid/
```

**Warning:** This downloads ~1.7TB of data. Not recommended unless you have:
- Fast internet
- Lots of disk space
- Time to wait

## Method 3: Use Local TFDS Cache Directory

If you've already started downloading (even partially), use the cache:

```bash
# Check what's in the cache
ls -lh ~/.cache/tensorflow_datasets/droid/

# Use the cache directory
python download_block_episodes.py \
    --episode-ids block_episode_ids.json \
    --max-episodes 5 \
    --tfds-dir ~/.cache/tensorflow_datasets/droid/ \
    --output-dir data/block_episodes \
    --annotations droid_language_annotations.json
```

## Method 4: Download Using TFDS Download Manager

Use TFDS's built-in download with better retry logic:

```python
import tensorflow_datasets as tfds

# Configure download with retries
builder = tfds.builder("droid", data_dir="gs://gresearch/robotics")

# Download with configuration
builder.download_and_prepare(
    download_config=tfds.download.DownloadConfig(
        num_parallel_calls=1,  # Reduce parallel downloads
        verify_ssl=True,
    )
)

# Then load
ds = builder.as_dataset(split="train")
```

## Method 5: Use a Different Network/VPN

If network issues persist:

1. **Try a different network** (if possible)
2. **Use a VPN** to a different location
3. **Try at a different time** (GCS may have less load)
4. **Use a machine with better connectivity**

## Quick Check: What's Already Cached?

```bash
# Check TFDS cache
du -sh ~/.cache/tensorflow_datasets/droid/ 2>/dev/null || echo "No cache yet"

# List cached files
find ~/.cache/tensorflow_datasets/droid/ -name "*.tfrecord" 2>/dev/null | head -10
```

## Recommended Approach

Given the network timeout issues, I recommend:

1. **Try the script again later** (network issues may be temporary)
2. **Use Method 1 (gsutil)** if you need specific episodes
3. **Use Method 3** if TFDS has already cached some data

Let me create a helper script for Method 1.


