# Quick start

```bash
# 0. Browse available foundation models and their registry IDs
bioaccx registry

# 1. Copy and edit the example config
cp example_config.yaml my_config.yaml

# 2. Validate config without running training
bioaccx validate my_config.yaml

# 3. Export the dataset as chunked WAV files (no model needed)
bioaccx dataset my_config.yaml

# 4. Compute the embedding database + UMAP (no training)
bioaccx embeddings my_config.yaml

# 5. Train and export
bioaccx train my_config.yaml

# 6. Merge an existing classifier head with a backbone (without training)
bioaccx merge my_config.yaml

# 7. Extract the head from a full custom BirdNET-Analyzer TFLite model (without training)
bioaccx extract-head my_config.yaml
```
