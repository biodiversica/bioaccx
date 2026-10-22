<!-- translated-from: docs/en/quickstart.md@ede516fe236e -->
# Primeiros passos

```bash
# 0. Veja os modelos base disponíveis e seus IDs de registro
bioaccx registry

# 1. Copie e edite a configuração de exemplo
cp example_config.yaml my_config.yaml

# 2. Valide a configuração sem executar o treinamento
bioaccx validate my_config.yaml

# 3. Exporte o dataset como arquivos WAV segmentados (sem o modelo)
bioaccx dataset my_config.yaml

# 4. Calcule o banco de embeddings e a projeção UMAP (sem treinamento)
bioaccx embeddings my_config.yaml

# 5. Treine e exporte
bioaccx train my_config.yaml

# 6. Funda um classifier head existente com um backbone (sem treinamento)
bioaccx merge my_config.yaml

# 7. Extraia o head de um modelo TFLite completo do BirdNET-Analyzer (sem treinamento)
bioaccx extract-head my_config.yaml
```
