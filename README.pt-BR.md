<!-- translated-from: README.md@4d87cc869ba7 -->
# bioaccx

**BIOAcoustic Custom Classifier eXchange** — uma ferramenta Python de linha de comando para treinar e compartilhar classificadores bioacústicos customizados sobre modelos base pré-treinados como o [BirdNET](https://birdnet.cornell.edu/) e o [Perch](https://www.kaggle.com/models/google/bird-vocalization-classifier).

A ideia central é separar de forma limpa o **backbone** e o **classifier head**:

- O backbone (modelo base) é um codificador de áudio grande e de uso geral, compartilhado pela comunidade.
- O head é um classificador pequeno, específico da sua tarefa, treinado nos seus dados rotulados — alguns poucos kilobytes.

Essa separação permite dois fluxos complementares:

**Compartilhar heads leves.** Como classificadores customizados são geralmente pequenos perto do backbone, você pode publicar e distribuir seu classificador sem levar junto o modelo inteiro. Quem já tem o mesmo backbone carrega o seu classificador e roda a inferência na hora.

**Publicar como um arquivo único.** Quando você precisa de um modelo autocontido — para dispositivos de borda, APIs na nuvem ou ferramentas de terceiros — o bioaccx funde backbone e head em um único arquivo ONNX ou TFLite, com uma entrada de áudio e uma saída de pontuações por classe.

**Pular o reprocessamento quando você já tem os embeddings.** Se você ou sua equipe já rodaram o backbone e guardaram os vetores de embedding, dá para apontar o bioaccx para esse cache e treinar ou avaliar o head direto — sem processar áudio, sem tempo de GPU, sem espera.

**Montar datasets de várias fontes em uma configuração só.** O bioaccx monta os dados de treino a partir de gravações locais, observações do [iNaturalist](https://www.inaturalist.org/), gravações do [Xeno-canto](https://xeno-canto.org/) e projetos do [Arbimon](https://arbimon.org/) — tudo misturado em uma tabela, ou combinado com um diretório local. O áudio remoto é baixado e cacheado automaticamente, então as execuções seguintes não tocam a rede.

O bioaccx cuida do caminho inteiro, do áudio bruto ao modelo exportado, guiado por um único arquivo de configuração.

> 🇬🇧 [Read in English](README.md)

---

## Como funciona

```
Arquivos de áudio  →  Modelo base (backbone)  →  Embeddings
                                                      ↓
                              Classifier head Keras ou sklearn
                                                      ↓
                             ┌────────────────────────────────────────┐
                             │  backbone + head fundidos │ só o head  │
                             │    (arquivo único)        │(compartilhável)│
                             └────────────────────────────────────────┘
```

1. **Modelo base** — uma versão backbone de um modelo bioacústico extrai vetores de características ricos de janelas de áudio bruto. Formatos suportados: ONNX, TFLite, TF SavedModel (protobuf).
2. **Classifier head** — uma pequena MLP em Keras ou uma LogisticRegression do sklearn é treinada sobre esses embeddings com os seus dados rotulados.
3. **Exportação** — escolha entre:
   - **Modelo completo** (`output_type: full`): funde backbone + head em um único grafo ONNX ou TFLite. Inferência direta, sem dependências externas.
   - **Só o classificador (head)** (`output_type: head`): exporta apenas o classificador. Arquivo minúsculo, fácil de compartilhar; exige o backbone na hora da inferência.
   - **Ambos** (`output_type: both`): produz as duas variantes.

---

## Instalação

O bioaccx executa os modelos pelo ONNX Runtime, e **não há um padrão** — escolha
`cpu` ou `gpu` na instalação. Um `bioaccx` sem extras instala sem reclamar e
falha no momento em que tenta rodar um modelo, então na prática o extra não é
opcional.

```bash
uv tool install "bioaccx[cpu]"              # inferência na CPU
uv tool install "bioaccx[gpu]"              # CUDA 12
uv tool install "bioaccx[cpu,gui,umap]"     # com a GUI no navegador e os gráficos UMAP

uv add "bioaccx[cpu]"                       # ou como dependência de um projeto
```

Os extras vão **dentro dos colchetes** — o `uv tool install` não tem a opção
`--extra` — e o argumento inteiro precisa de aspas para o shell não expandi-los.

Veja [Instalação](docs/pt-BR/installation.md) para a tabela de extras, como
trabalhar no próprio bioaccx, aceleração por GPU com CUDA e quanto disso ocupa
em disco.

---

## Primeiros passos

```bash
# 0. Veja os modelos base disponíveis e seus IDs
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
```

O `bioaccx merge` e o `bioaccx extract-head` fazem os mesmos trabalhos sem
treinar — veja
[Merging, extracting and converting heads](docs/en/model-surgery.md).

---

## A GUI no navegador

O `bioaccx gui` abre um editor de configuração no navegador. Ele escreve os
mesmos arquivos YAML que todos os outros comandos leem — o terminal continua
sendo o lugar de onde as execuções partem.

```bash
uv tool install "bioaccx[cpu,gui]"      # a GUI precisa do extra [gui]

bioaccx gui                             # começa de um modelo em branco
bioaccx gui my_config.yaml              # abre uma configuração existente
bioaccx gui --lang pt-BR                # abre em português
```

O formulário é gerado a partir das dataclasses de configuração, então oferece
exatamente as keys que esta versão entende, com os valores padrão de verdade e
os textos de ajuda; o arquivo que ele escreve aparece ao lado enquanto você
digita, e seus comentários sobrevivem ao round-trip. A aba **modelos** lê o que
as execuções passadas escreveram — métricas, o mapa UMAP com o áudio por trás de
cada ponto, e duas execuções lado a lado. As execuções podem ser iniciadas pela
página, com o log chegando nela ao vivo.

Veja [A GUI no navegador](docs/pt-BR/gui.md) para tudo isso, incluindo os
[idiomas da interface](docs/pt-BR/gui.md#idiomas-da-interface) e como servir o
editor para outra máquina.

---

## Configuração

Tudo o que uma execução faz é descrito por um único arquivo YAML: o
[`example_config.yaml`](example_config.yaml) é um ponto de partida que funciona,
[Configuration](docs/en/configuration.md) explica seção por seção (em inglês), e
a [Referência de configuração](docs/pt-BR/config-reference.md) lista todas as
keys com seus valores padrão.

---

## Documentação

A documentação completa está em [`docs/pt-BR/`](docs/pt-BR/). Esta página cobre
o que o bioaccx é, como instalá-lo e como tirar dele uma primeira execução.

| Página | Conteúdo |
|---|---|
| [Instalação](docs/pt-BR/installation.md) | Formas de instalar, a tabela de extras, aceleração por GPU, espaço em disco |
| [Primeiros passos](docs/pt-BR/quickstart.md) | Os comandos de uma primeira execução, em ordem |
| [A GUI no navegador](docs/pt-BR/gui.md) | `bioaccx gui` — editor de configuração, explorador de resultados, idiomas |
| [Referência de configuração](docs/pt-BR/config-reference.md) | Todas as keys, com valor padrão e descrição |
| [Configuration](docs/en/configuration.md) 🇬🇧 | O arquivo de configuração seção por seção, com exemplos |
| [Datasets](docs/en/dataset-modes.md) 🇬🇧 | Formatos de rótulo, divisão train/test, acréscimo, múltiplas fontes |
| [Remote sound sources](docs/en/remote-sources.md) 🇬🇧 | iNaturalist, Xeno-canto e Arbimon em uma tabela só |
| [Augmentation and windowing](docs/en/augmentation.md) 🇬🇧 | Ruído, deslocamento aleatório, janelas e sobreposição |
| [Embeddings, cache and UMAP](docs/en/embeddings-umap.md) 🇬🇧 | O banco de embeddings, `bioaccx embeddings`, UMAP e KMeans/NMI |
| [Outputs](docs/en/outputs.md) 🇬🇧 | Estrutura de diretórios, metadados, precisão, excluir rótulos |
| [Merging, extracting and converting heads](docs/en/model-surgery.md) 🇬🇧 | `merge`, `extract-head`, `convert-head` |
| [Foundation model registry](docs/en/registry.md) 🇬🇧 | IDs do registro, modelos suportados, adicionar um novo |
| [Python API](docs/en/python-api.md) 🇬🇧 | Usar o bioaccx como biblioteca |
| [CLI reference](docs/en/cli.md) 🇬🇧 | Todos os comandos e opções |

As páginas marcadas com 🇬🇧 ainda não foram traduzidas e estão em inglês.

As referências de configuração e da CLI são geradas a partir do código pelo
`tools/gen_docs.py`, então não têm como ficar para trás dele — veja
[`docs/README.md`](docs/README.md).
