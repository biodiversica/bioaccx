<!-- translated-from: docs/en/quickstart.md@e73dea3a4221 -->
# Primeiros passos

```bash
# 0. Veja os modelos base disponíveis e seus IDs de registro
bioaccx registry

# 1. Copie e edite a configuração de exemplo
cp example_config.yaml my_config.yaml

# 2. Valide a configuração sem rodar o treinamento
bioaccx validate my_config.yaml

# 3. Exporte o dataset como arquivos WAV em trechos (sem precisar do modelo)
bioaccx dataset my_config.yaml

# 4. Calcule o banco de embeddings + UMAP (sem treinar)
bioaccx embeddings my_config.yaml

# 5. Treine e exporte
bioaccx train my_config.yaml

# 6. Junte um classifier head já existente com um backbone (sem treinar)
bioaccx merge my_config.yaml

# 7. Extraia o head de um modelo TFLite completo do BirdNET-Analyzer (sem treinar)
bioaccx extract-head my_config.yaml
```
