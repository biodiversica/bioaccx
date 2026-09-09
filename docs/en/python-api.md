# Python API

bioaccx can also be used as a library:

```python
from bioaccx.config import load_config
from bioaccx.train import run

cfg = load_config("my_config.yaml")
outputs = run(cfg)
print(outputs)
# {'dataset_list': '...', 'keras_onnx_head': '...', 'model_info': '...', ...}
```

Constructing a config programmatically:

```python
from bioaccx.config import (
    BioaccxConfig, FoundationModelConfig, DatasetConfig,
    TrainingConfig, KerasConfig, OutputConfig,
)

cfg = BioaccxConfig(
    foundation_model=FoundationModelConfig(
        name="birdnet",
        version="2.4",
        format="onnx",
        source="local",
        path="/models/birdnet_headless.onnx",
        sample_rate=48000,
        window_seconds=3.0,
        input_name="INPUT",
        embedding_size=1024,
    ),
    dataset=DatasetConfig(
        data_dir="/data/birds",
        label_mode="file_per_label",
        overlap=0.5,
        embedding_workers=8,
    ),
    training=TrainingConfig(
        classifier="both",
        keras=KerasConfig(
            epochs=100,
            hidden_units=512,
            normalize_embeddings=True,
            focal_loss=True,
        ),
    ),
    output=OutputConfig(
        output_path="./models",
        model_name="bird_classifier",
        model_version="1.0",
        output_type="both",
        exclude_labels=["background"],
    ),
)

from bioaccx.train import run
outputs = run(cfg)
```
