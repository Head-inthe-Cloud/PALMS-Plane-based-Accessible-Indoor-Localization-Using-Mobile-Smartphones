# Guide: Uploading PALMS Dataset to HuggingFace

## Quick Answers

### Is it free?
**Yes!** HuggingFace offers free hosting for:
- **Public datasets**: Unlimited storage (free)
- **Private datasets**: Up to 100GB free (your dataset is ~38GB, so you're within the limit)

### Do you need to zip?
**Not required, but recommended** for:
- Faster uploads
- Better organization
- Reduced storage costs
- Easier version control

### Best compression method?
For your dataset structure (images, JSON, numpy arrays), use **TAR.GZ** or **ZIP**:
- **TAR.GZ**: Better compression ratio, standard for datasets
- **ZIP**: More universal, easier to extract on Windows

---

## Recommended Approach

### Option 1: Upload as Separate Components (Recommended)

Split your dataset into logical components for easier management:

```bash
# Create compressed archives for each component
cd ../datasets

# 1. Main dataset (largest component)
tar -czf palms_main_dataset.tar.gz main_dataset/

# 2. Panorama samples (smaller, for quick testing)
tar -czf palms_pano_samples.tar.gz pano_samples/

# 3. Maps (small CSV files - can upload directly)
# Keep maps/ as-is (small files, no compression needed)

# 4. Trajectories (moderate size)
tar -czf palms_trajectories.tar.gz trajectories/

# 5. Structured3D (if you have it)
# Note: Check Structured3D license before uploading
```

**Advantages:**
- Users can download only what they need
- Faster incremental updates
- Better organization

### Option 2: Single Archive (Simpler)

```bash
cd ../datasets
tar -czf palms_full_dataset.tar.gz .
```

**Advantages:**
- Single file to manage
- Simpler for users who want everything

---

## Step-by-Step Upload Process

### Method 1: Using HuggingFace Web Interface

1. **Create a HuggingFace account** (if you don't have one)
   - Go to https://huggingface.co/join

2. **Create a new dataset repository**
   - Click your profile → "New Dataset"
   - Name it (e.g., `palms-dataset` or `palms-indoor-localization`)
   - Choose **Public** (free, unlimited) or **Private** (free up to 100GB)
   - Click "Create repository"

3. **Upload files**
   - Go to "Files and versions" tab
   - Click "Add file" → "Upload files"
   - Drag and drop your compressed archives or folders
   - Wait for upload to complete (may take time for 38GB)

### Method 2: Using Python (Recommended for Large Datasets)

```python
from huggingface_hub import HfApi, login
import os

# 1. Login (will prompt for token)
login()

# 2. Initialize API
api = HfApi()

# 3. Upload folder or files
repo_id = "your_username/palms-dataset"  # Replace with your username

# Option A: Upload entire folder (will upload files individually)
api.upload_folder(
    folder_path="../datasets",
    repo_id=repo_id,
    repo_type="dataset",
    ignore_patterns=["*.DS_Store", "__pycache__", "*.pyc"],  # Exclude system files
)

# Option B: Upload specific compressed files
api.upload_file(
    path_or_fileobj="../datasets/palms_main_dataset.tar.gz",
    path_in_repo="palms_main_dataset.tar.gz",
    repo_id=repo_id,
    repo_type="dataset",
)
```

### Method 3: Using Git LFS (For Version Control)

```bash
# Install git-lfs if not already installed
# macOS: brew install git-lfs
# Linux: sudo apt-get install git-lfs

# Initialize git-lfs
git lfs install

# Clone your dataset repo
git clone https://huggingface.co/datasets/your_username/palms-dataset
cd palms-dataset

# Track large files
git lfs track "*.tar.gz"
git lfs track "*.npy"
git lfs track "*.png"

# Add and commit
git add .gitattributes
git add palms_main_dataset.tar.gz
git commit -m "Add main dataset"
git push
```

---

## Compression Best Practices

### For Your Dataset Structure

Your dataset contains:
- **Images** (PNG): Already compressed, but can benefit from TAR.GZ
- **JSON files**: Text-based, compresses well
- **NumPy arrays** (.npy): Binary, moderate compression
- **CSV files**: Text-based, compresses very well

### Recommended Compression Commands

```bash
# High compression (slower, smaller files)
tar -czf --best palms_main_dataset.tar.gz main_dataset/

# Balanced (recommended)
tar -czf palms_main_dataset.tar.gz main_dataset/

# Fast compression (faster, larger files)
tar -czf --fast palms_main_dataset.tar.gz main_dataset/

# For ZIP (more universal)
zip -r -9 palms_main_dataset.zip main_dataset/  # -9 = maximum compression
```

### Check Compression Ratio

```bash
# Before compression
du -sh main_dataset/

# After compression
du -sh palms_main_dataset.tar.gz

# Calculate ratio
echo "Compression ratio: $(echo "scale=2; $(du -sb main_dataset/ | cut -f1) / $(du -sb palms_main_dataset.tar.gz | cut -f1)" | bc)"
```

---

## Dataset Card Template

Create a `README.md` in your HuggingFace dataset repo:

```markdown
---
license: mit  # or your chosen license
task_categories:
- other
tags:
- indoor-localization
- computer-vision
- robotics
- accessibility
size_categories:
- 10K<n<100K
---

# PALMS Dataset

## Dataset Description

[Describe your dataset here]

## Dataset Structure

```
datasets/
├── main_dataset/        # Full recording sessions (38GB)
├── pano_samples/        # Panorama samples for quick testing
├── maps/                # Floor plan geometry files
├── trajectories/        # IMU-based odometry tracking data
└── structured3d/        # Structured3D synthetic scenes (if included)
```

## Usage

[Add usage instructions]

## Citation

[Add citation information]
```

---

## Important Considerations

### 1. Privacy & Ethics
- Your README mentions blurred humans - ensure compliance
- Consider adding a data use agreement
- Document any privacy considerations

### 2. License
- Choose an appropriate license (MIT, CC-BY, etc.)
- Specify usage terms clearly

### 3. File Size Limits
- HuggingFace has a **50GB per file** limit
- If any single file exceeds 50GB, split it or use Git LFS

### 4. Upload Time
- 38GB upload will take significant time (hours to days depending on connection)
- Consider uploading during off-peak hours
- Use Python API for resumable uploads

### 5. Version Control
- HuggingFace supports dataset versioning
- Tag releases for reproducibility

---

## Quick Start Script

Save this as `upload_dataset.sh`:

```bash
#!/bin/bash

# Configuration
REPO_ID="your_username/palms-dataset"  # Change this
DATASET_DIR="../datasets"
COMPRESS=true

cd "$DATASET_DIR"

if [ "$COMPRESS" = true ]; then
    echo "Compressing datasets..."
    tar -czf palms_main_dataset.tar.gz main_dataset/
    tar -czf palms_pano_samples.tar.gz pano_samples/
    tar -czf palms_trajectories.tar.gz trajectories/
    echo "Compression complete!"
fi

echo "Uploading to HuggingFace..."
python3 << EOF
from huggingface_hub import HfApi, login

login()  # Will prompt for token
api = HfApi()

# Upload compressed files
for file in ["palms_main_dataset.tar.gz", "palms_pano_samples.tar.gz", "palms_trajectories.tar.gz"]:
    if os.path.exists(file):
        print(f"Uploading {file}...")
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=file,
            repo_id="$REPO_ID",
            repo_type="dataset",
        )

# Upload maps directly (small files)
api.upload_folder(
    folder_path="maps",
    path_in_repo="maps",
    repo_id="$REPO_ID",
    repo_type="dataset",
)

print("Upload complete!")
EOF
```

Make it executable:
```bash
chmod +x upload_dataset.sh
```

---

## Troubleshooting

### Upload Fails/Timeout
- Use Python API with retry logic
- Upload in smaller chunks
- Check your internet connection stability

### File Too Large
- Split into multiple archives
- Use Git LFS for version control

### Authentication Issues
- Generate a token at https://huggingface.co/settings/tokens
- Use `huggingface-cli login` command

---

## Next Steps

1. **Choose your approach** (separate components vs. single archive)
2. **Compress your data** using the commands above
3. **Create HuggingFace dataset repository**
4. **Upload using your preferred method**
5. **Create a comprehensive dataset card** (README.md)
6. **Test download** to ensure everything works

Good luck with your dataset release! 🚀

