# Google Cloud Authentication Guide

This guide explains how to install and authenticate with Google Cloud to access the DROID dataset.

## Step 1: Install Google Cloud SDK

### On Linux/WSL (Your System)

```bash
# Download and install Google Cloud SDK
curl https://sdk.cloud.google.com | bash

# Or using snap (if available)
sudo snap install google-cloud-cli --classic

# Or download directly:
# 1. Download from: https://cloud.google.com/sdk/docs/install
# 2. Extract and run: ./install.sh
```

### Alternative: Install via Package Manager

```bash
# Add the Google Cloud SDK repository
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Import the Google Cloud Platform public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Update and install
sudo apt-get update && sudo apt-get install google-cloud-cli
```

### For macOS

```bash
# Using Homebrew
brew install --cask google-cloud-sdk
```

## Step 2: Authenticate with Google Cloud

Once Google Cloud SDK is installed:

```bash
# Authenticate for application default credentials
gcloud auth application-default login

# This will:
# 1. Open a browser window
# 2. Ask you to sign in with your Google account
# 3. Grant permissions
# 4. Save credentials locally
```

**Note:** The DROID dataset is **publicly accessible**, so you don't need:
- A Google Cloud project
- A paid account
- Special permissions

You just need a Google account to authenticate.

## Step 3: Verify Authentication

```bash
# Check if authentication worked
gcloud auth application-default print-access-token

# Should print a long token, not an error
```

## Step 4: (Optional) Set Default Project

This is optional, but can help avoid warnings:

```bash
# List available projects (if any)
gcloud projects list

# Set default project (if you have one)
gcloud config set project YOUR_PROJECT_ID

# Or use the public research project (recommended)
gcloud config set project gresearch-public
```

## Alternative: Using Service Account Key (Advanced)

If you prefer not to use browser-based authentication:

1. **Create a service account** (in Google Cloud Console)
2. **Download the JSON key**
3. **Set environment variable:**

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

## Troubleshooting

### "gcloud: command not found"

**Solution:** Install Google Cloud SDK (see Step 1 above)

After installation, you may need to:
```bash
# Restart your terminal, or source the installation
source ~/.bashrc
# or
source ~/.zshrc
```

### "Access Denied" or "Permission Denied"

**Solution:** The dataset is public, but you still need to authenticate:
```bash
gcloud auth application-default login
```

### "Could not locate credentials"

**Solution:** Re-authenticate:
```bash
# Remove old credentials
rm -rf ~/.config/gcloud/

# Authenticate again
gcloud auth application-default login
```

### Browser Won't Open (Headless/Remote Server)

If you're on a remote server without a browser:

```bash
# Use a device code flow instead
gcloud auth application-default login --no-browser

# This will give you a URL and code to enter on another device
```

## Testing Access

After authentication, test if you can access the dataset:

```bash
# Test with a simple Python script
python -c "
import tensorflow_datasets as tfds
ds = tfds.load('droid', data_dir='gs://gresearch/robotics', split='train')
print('✅ Successfully connected to DROID dataset!')
for i, episode in enumerate(ds.take(1)):
    print(f'✅ Loaded episode {i+1}')
"
```

## Quick Command Reference

```bash
# Authenticate
gcloud auth application-default login

# Check authentication
gcloud auth application-default print-access-token

# List authenticated accounts
gcloud auth list

# Revoke credentials (if needed)
gcloud auth application-default revoke

# Re-authenticate
gcloud auth application-default login
```

## What This Enables

Once authenticated, you can:
- ✅ Access DROID dataset from `gs://gresearch/robotics`
- ✅ Run `download_block_episodes.py` without errors
- ✅ Stream episodes from Google Cloud Storage
- ✅ Use TensorFlow Datasets with DROID

The authentication is stored locally and persists across sessions until you revoke it.


