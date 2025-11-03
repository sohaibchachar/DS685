# Dataset Selection Note

## Issue: droid_100 has limited episodes

The `droid_100` dataset only contains ~100 episodes, but your target list has 172 episodes. This means:
- **160 out of 172 target episodes are in the mapping** (episode_id_to_path.json)
- **But only a few of those 172 episodes are actually in the droid_100 subset**

## Solution: Use Full Dataset

To find all 172 episodes, you need to use the **full DROID dataset** instead of the subset:

```bash
# Use full dataset (warning: very large!)
python download_specific_blocks.py --dataset-name droid --max-episodes 5
```

## Dataset Options

- `--dataset-name droid_100`: Small subset (~100 episodes) - fast but limited
- `--dataset-name droid`: Full dataset (many episodes) - slow but complete

## Recommendation

For testing, start with a smaller subset:
```bash
python download_specific_blocks.py --dataset-name droid_100 --max-episodes 2
```

Once you confirm it works, switch to full dataset for all episodes:
```bash
python download_specific_blocks.py --dataset-name droid
```

Note: The full dataset is very large and will take significantly longer to process.


