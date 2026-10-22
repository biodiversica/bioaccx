<!-- translated-from: README.md@f0def0aac48f -->
# bioaccx

**BIOAcoustic Custom Classifier eXchange** é uma ferramenta Python de linha de comando para treinar e compartilhar classificadores bioacústicos personalizados, construídos sobre modelos base pré-treinados como o [BirdNET](https://birdnet.cornell.edu/) e o [Perch](https://www.kaggle.com/models/google/bird-vocalization-classifier).

O bioaccx baseia-se em uma separação clara entre o **backbone** e o **classifier head**:

- O backbone (modelo base) é um codificador de áudio grande e de uso geral, compartilhado pela comunidade.
- O head é um classificador pequeno, específico da tarefa, treinado com os dados rotulados, e ocupa normalmente apenas alguns kilobytes.

Essa separação possibilita os seguintes fluxos de trabalho:

**Compartilhamento de heads leves.** Como o head é muito pequeno em comparação com o backbone, um classificador personalizado pode ser publicado e distribuído sem o modelo completo. Quem já dispõe do mesmo backbone pode carregar o head e executar a inferência imediatamente.

**Implantação como arquivo único.** Quando é necessário um modelo autocontido, por exemplo em dispositivos de borda, APIs na nuvem ou ferramentas de terceiros, o bioaccx pode fundir o backbone e o head em um único arquivo ONNX ou TFLite, com uma entrada de áudio e uma saída de pontuações por classe.

**Reaproveitamento de embeddings existentes.** Se o backbone já foi executado e os vetores de embedding resultantes foram armazenados, o bioaccx pode ler esse cache e treinar ou avaliar o head diretamente, sem processamento de áudio nem uso de GPU.

**Montagem de datasets de várias fontes em uma única configuração.** O bioaccx reúne dados de treino a partir de gravações locais, observações do [iNaturalist](https://www.inaturalist.org/), gravações do [Xeno-canto](https://xeno-canto.org/) e projetos do [Arbimon](https://arbimon.org/), listados em uma única tabela ou combinados com um diretório local. O áudio remoto é baixado e armazenado em cache automaticamente, de modo que as execuções seguintes não acessam a rede.

O bioaccx abrange todo o pipeline, do áudio bruto ao modelo exportado, e é controlado por um único arquivo de configuração.

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

1. **Modelo base:** o backbone de um modelo bioacústico extrai vetores de características de janelas de áudio bruto. Os formatos suportados são ONNX, TFLite e TF SavedModel (protobuf).
2. **Classifier head:** uma pequena MLP em Keras ou uma LogisticRegression do scikit-learn é treinada sobre esses embeddings com os dados rotulados.
3. **Exportação:** é gerada uma das seguintes variantes:
   - **Modelo completo** (`output_type: full`): o backbone e o head fundidos em um único grafo ONNX ou TFLite, executável sem dependências externas.
   - **Apenas o classificador (head)** (`output_type: head`): somente o classificador. O arquivo é pequeno e fácil de compartilhar, mas o backbone é necessário no momento da inferência.
   - **Ambos** (`output_type: both`): as duas variantes.

---

## Instalação

**Sem Python.** O instalador prepara tudo o que é necessário e adiciona um atalho
**bioaccx** que abre a [GUI no navegador](#a-gui-no-navegador). Executá-lo
novamente atualiza a instalação.

No Windows, abra o PowerShell e execute:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://github.com/biodiversica/bioaccx/releases/latest/download/install.ps1 | iex"
```

No macOS (Apple Silicon) e no Linux, abra um terminal e execute:

```sh
curl -LsSf https://github.com/biodiversica/bioaccx/releases/latest/download/install.sh | sh
```

**Com uv ou pip.** O bioaccx executa os modelos pelo ONNX Runtime, e **nenhum
runtime é instalado por padrão**: escolha `cpu` ou `gpu` na instalação. A
instalação de `bioaccx` sem extras é concluída, mas a primeira tentativa de
executar um modelo falha; na prática, portanto, o extra é obrigatório.

```bash
uv tool install --python 3.12 "bioaccx[cpu]"            # inferência na CPU
uv tool install --python 3.12 "bioaccx[gpu]"            # CUDA 12
uv tool install --python 3.12 "bioaccx[cpu,gui,umap]"   # com a GUI no navegador e os gráficos UMAP

uv add "bioaccx[cpu]"                                   # ou como dependência de um projeto
```

Os extras são indicados **dentro dos colchetes**, pois o `uv tool install` não
possui a opção `--extra`, e o argumento inteiro deve estar entre aspas para que
o shell não os expanda. O `--python` impede que o uv selecione uma versão do
Python mais recente do que a suportada pelo TensorFlow.

A página [Instalação](docs/pt-BR/installation.md) descreve o que o instalador
faz, os extras disponíveis, os sistemas suportados, a configuração para
desenvolvimento, a aceleração por GPU com CUDA e o espaço em disco necessário.

---

## Primeiros passos

```bash
# 0. Veja os modelos base disponíveis e seus IDs
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
```

O `bioaccx merge` e o `bioaccx extract-head` geram modelos sem treinamento;
consulte
[Merging, extracting and converting heads](docs/en/model-surgery.md).

---

## A GUI no navegador

O `bioaccx gui` abre um editor de configuração no navegador. Ele grava os mesmos
arquivos YAML lidos por todos os outros comandos, e as execuções continuam
podendo ser iniciadas pelo terminal.

```bash
uv tool install "bioaccx[cpu,gui]"      # a GUI precisa do extra [gui]

bioaccx gui                             # começa a partir de um modelo de configuração
bioaccx gui my_config.yaml              # abre uma configuração existente
bioaccx gui --lang pt-BR                # abre em português
```

O formulário é gerado a partir das dataclasses de configuração e, portanto,
oferece exatamente as keys suportadas por esta versão, com os valores padrão
reais e os textos de ajuda. O arquivo gravado é exibido ao lado do formulário
durante a edição, e os comentários do arquivo são preservados. A aba
**Execução** inicia um comando sobre a configuração e transmite o log para a
página. A aba **Análise** apresenta os resultados de execuções anteriores:
métricas, o mapa UMAP com o áudio de cada ponto e a comparação lado a lado de
duas execuções.

A página [A GUI no navegador](docs/pt-BR/gui.md) descreve o editor por completo,
incluindo os [idiomas da interface](docs/pt-BR/gui.md#idiomas-da-interface) e o
acesso a partir de outra máquina.

---

## Configuração

Cada execução é descrita integralmente por um único arquivo YAML. O
[`example_config.yaml`](example_config.yaml) é um ponto de partida funcional, a
página [Configuration](docs/en/configuration.md) o explica seção por seção (em
inglês), e a [Referência de configuração](docs/pt-BR/config-reference.md) lista
todas as keys com seus valores padrão.

---

## Documentação

A documentação completa está em [`docs/pt-BR/`](docs/pt-BR/). Esta página
descreve o que é o bioaccx, como instalá-lo e como realizar uma primeira
execução.

| Página | Conteúdo |
|---|---|
| [Instalação](docs/pt-BR/installation.md) | Formas de instalar, a tabela de extras, aceleração por GPU, espaço em disco |
| [Primeiros passos](docs/pt-BR/quickstart.md) | Os comandos de uma primeira execução, em ordem |
| [A GUI no navegador](docs/pt-BR/gui.md) | `bioaccx gui`: editor de configuração, explorador de resultados, idiomas |
| [Referência de configuração](docs/pt-BR/config-reference.md) | Todas as keys, com valor padrão e descrição |
| [Configuration](docs/en/configuration.md) 🇬🇧 | O arquivo de configuração seção por seção, com exemplos |
| [Datasets](docs/en/dataset-modes.md) 🇬🇧 | Formatos de rótulo, divisão train/test, acréscimo, múltiplas fontes |
| [Remote sound sources](docs/en/remote-sources.md) 🇬🇧 | iNaturalist, Xeno-canto e Arbimon em uma única tabela |
| [Augmentation and windowing](docs/en/augmentation.md) 🇬🇧 | Ruído, deslocamento aleatório, janelas e sobreposição |
| [Embeddings, cache and UMAP](docs/en/embeddings-umap.md) 🇬🇧 | O banco de embeddings, `bioaccx embeddings`, UMAP e KMeans/NMI |
| [Outputs](docs/en/outputs.md) 🇬🇧 | Estrutura de diretórios, metadados, precisão, exclusão de rótulos |
| [Merging, extracting and converting heads](docs/en/model-surgery.md) 🇬🇧 | `merge`, `extract-head`, `convert-head` |
| [Foundation model registry](docs/en/registry.md) 🇬🇧 | IDs do registro, modelos suportados, inclusão de um novo modelo |
| [Python API](docs/en/python-api.md) 🇬🇧 | Uso do bioaccx como biblioteca |
| [CLI reference](docs/en/cli.md) 🇬🇧 | Todos os comandos e opções |

As páginas marcadas com 🇬🇧 ainda não foram traduzidas e estão em inglês.

As referências de configuração e da CLI são geradas a partir do código pelo
`tools/gen_docs.py` e, portanto, estão sempre de acordo com ele; consulte
[`docs/README.md`](docs/README.md).

---

## Licença

[MIT](LICENSE)
