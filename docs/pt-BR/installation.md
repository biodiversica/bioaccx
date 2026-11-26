<!-- translated-from: docs/en/installation.md@0aa8ed5d5df7 -->
# Instalação

*[English](../en/installation.md) · Português (Brasil)*

## O instalador (sem necessidade de Python)

No Windows, abra o PowerShell e execute:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://github.com/biodiversica/bioaccx/releases/latest/download/install.ps1 | Out-String | iex"
```

No macOS (Apple Silicon) e no Linux, abra um terminal e execute:

```sh
curl -LsSf https://github.com/biodiversica/bioaccx/releases/latest/download/install.sh | sh
```

O instalador executa as seguintes etapas:

1. Instala o [uv](https://docs.astral.sh/uv/) em `~/.local/bin`, caso ainda não
   esteja presente. O uv baixa o próprio Python, de modo que nenhuma instalação
   prévia é necessária, e qualquer instalação de Python existente permanece
   inalterada.
2. Executa `uv tool install --python 3.12 "bioaccx[cpu,gui,umap]"`, que instala o
   runtime de CPU, a GUI no navegador e os gráficos UMAP, e adiciona o `bioaccx`
   ao PATH.
3. Cria uma pasta de trabalho, `~/bioaccx`, na qual a GUI é iniciada. As
   configurações salvas e os modelos treinados são armazenados nela, a menos que
   outro local seja escolhido.
4. Adiciona um atalho **bioaccx**, com o logotipo do bioaccx, que executa
   `bioaccx gui` nessa pasta: na Área de Trabalho e no menu Iniciar, no Windows; `bioaccx.command` na Mesa, no
   macOS; e uma entrada no menu de aplicativos, no Linux. O atalho abre uma
   janela de terminal ao lado do navegador. Essa janela executa o servidor, e
   fechá-la (ou pressionar Ctrl+C) encerra o servidor.

Antes de qualquer uma dessas etapas, o instalador lista o que vai fazer e pede
confirmação. Executá-lo novamente atualiza o bioaccx. Variáveis de ambiente
alteram os padrões: `BIOACCX_PACKAGE` (o pacote a instalar, por exemplo
`bioaccx[gpu,gui]`), `BIOACCX_WORKDIR` (a pasta de trabalho) e `BIOACCX_YES=1`
(prosseguir sem perguntar, para scripts e CI; sem ela, o instalador se recusa a
executar onde não há um terminal em que perguntar).

### Desinstalação

O mesmo instalador desinstala o bioaccx quando executado com
`BIOACCX_UNINSTALL=1`. Ele também lista o que vai remover e pede confirmação
antes:

```powershell
powershell -ExecutionPolicy ByPass -c '$env:BIOACCX_UNINSTALL=1; irm https://github.com/biodiversica/bioaccx/releases/latest/download/install.ps1 | Out-String | iex'
```

```sh
curl -LsSf https://github.com/biodiversica/bioaccx/releases/latest/download/install.sh | BIOACCX_UNINSTALL=1 sh
```

`BIOACCX_UNINSTALL=1` remove o bioaccx e o seu atalho. `BIOACCX_UNINSTALL=all`
também remove os backbones BirdNET e Perch que o bioaccx baixou para o cache do
Hugging Face, e nenhum outro modelo armazenado ali. No Windows, feche a GUI
antes: o Windows não consegue remover um programa em execução.

**O que a desinstalação deixa no computador.** O ambiente próprio do bioaccx,
com todos os pacotes Python instalados para ele, é removido, junto com o comando
`bioaccx`. O que foi usado para instalá-lo permanece, mesmo com `=all`, pois o
desinstalador não tem como saber se outro programa o utiliza:

| Permanece | Onde | Motivo |
|---|---|---|
| A pasta de trabalho | `~/bioaccx` | Contém as configurações salvas e os modelos treinados. |
| uv | `~/.local/bin/uv` e `uvx` (`.exe` no Windows) | Outros programas podem utilizá-lo. |
| Python 3.12 | `uv python dir` | O Python próprio do uv, que outras ferramentas do uv podem utilizar. Se já havia um Python 3.12 instalado, o uv o utilizou e não baixou nenhum. |
| O cache de downloads do uv | `uv cache dir` | Compartilhado por tudo o que o uv instala. Contém os pacotes baixados, sendo o TensorFlow de longe o maior, e pode ocupar mais de 2 GB. |
| Modelos do Kaggle ou de outra origem | seus próprios caches | Apenas os backbones do Hugging Face do registro são removidos, com `=all`. |

Se nenhum outro programa no computador utiliza o uv, o restante pode ser
removido manualmente após a desinstalação:

```bash
uv cache clean                # todos os pacotes baixados
uv python uninstall 3.12      # o Python do uv, se ele baixou um
```

e, depois, o próprio uv, excluindo `uv` e `uvx` de `~/.local/bin`.

Nenhum dos comandos de cache pode danificar um programa instalado: um ambiente
não lê o cache depois de instalado, e cada comando aguarda enquanto outro uv
estiver usando o cache. Eles diferem no que excluem:

- `uv cache prune` exclui apenas o que não é mais referenciado, como entradas
  deixadas por versões antigas do uv. É o mais cauteloso, mas os pacotes
  baixados continuam referenciados; por isso, após a desinstalação, libera
  pouco espaço.
- `uv cache clean PACOTE...` exclui apenas os pacotes indicados. `uv cache clean
  tensorflow tensorflow-cpu` libera a maior parte do espaço ocupado pelo
  bioaccx, a menos que outras ferramentas também usem o TensorFlow.
- `uv cache clean` exclui o cache inteiro, de todos os programas. É o que mais
  libera espaço, e tudo o que for instalado com o uv depois volta a baixar os
  seus pacotes.

## Sistemas suportados

| Sistema | |
|---|---|
| Linux x86-64 e ARM64 | ✓ |
| Windows x86-64 | ✓ |
| macOS em Apple Silicon (M1 em diante) | ✓ |
| macOS em Intel | ✗ (o TensorFlow não publica versões para essa plataforma) |

É necessário o Python 3.12 ou 3.13, as versões atualmente suportadas pelo
TensorFlow. O instalador e os comandos abaixo fixam a versão 3.12, para que um
Python mais recente na máquina nunca seja selecionado por acidente.

## Como ferramenta de linha de comando

Esta é a instalação usual para quem já está familiarizado com o terminal. O
bioaccx executa os modelos pelo ONNX Runtime, e **nenhum runtime é instalado por
padrão**: escolha `cpu` ou `gpu` na instalação. A instalação de `bioaccx` sem
extras é concluída, mas a primeira tentativa de executar um modelo falha; na
prática, portanto, o extra é obrigatório.

O `uv tool install` instala o bioaccx em um ambiente isolado e adiciona o
`bioaccx` ao PATH:

```bash
uv tool install --python 3.12 "bioaccx[cpu]"            # inferência na CPU
uv tool install --python 3.12 "bioaccx[gpu]"            # CUDA 12 (consulte a seção de GPU abaixo)
uv tool install --python 3.12 "bioaccx[cpu,gui,umap]"   # com a GUI no navegador e os gráficos UMAP
```

O `uv tool install` **não possui a opção `--extra`**: os extras são indicados
dentro dos colchetes, e o argumento inteiro deve estar entre aspas para que o
shell não os expanda.

```bash
uv tool install --python 3.12 "/caminho/do/bioaccx[cpu]"   # a partir de um clone local
uv tool upgrade bioaccx                                    # atualização
uv tool uninstall bioaccx                                  # desinstalação
```

## Em um projeto

```bash
uv add "bioaccx[cpu]"                       # como dependência
pip install "bioaccx[cpu]"                  # ou com o pip
```

## Configuração para desenvolvimento

```bash
git clone git@github.com:biodiversica/bioaccx.git
cd bioaccx
uv sync --extra cpu                         # ou --extra gpu
uv sync --extra cpu --extra gui --extra umap
uv run bioaccx --help
uv run pytest
```

O `uv sync --extra X` **substitui** o conjunto de extras ativo em vez de
acrescentar a ele, portanto todos os extras necessários devem ser indicados a
cada execução. Sincronizar apenas com `--extra gui` remove o `onnxruntime` e o
`umap-learn`.

### Publicação de uma versão

Defina a nova versão no `pyproject.toml` e no `bioaccx/__init__.py`, faça o
commit da alteração e, em seguida, crie e envie a tag:

```bash
git tag v0.2.0 && git push origin v0.2.0
```

A tag aciona o `.github/workflows/release.yml`. Os testes e os instaladores são
executados no Linux, no macOS e no Windows, o wheel é enviado ao PyPI, e uma
release em **rascunho** é criada no GitHub com os instaladores anexados e as
notas de `.github/release-notes.md`. Depois de redigidos os destaques, a release
deve ser publicada, pois os comandos de instalação baixam da última release
*publicada*.

## Extras

| Extra | Adiciona | Necessário para |
|---|---|---|
| `cpu` | `onnxruntime` | Executar qualquer modelo na CPU. Escolha este ou o `gpu`. |
| `gpu` | `onnxruntime-gpu` | Executar com CUDA 12. Escolha este ou o `cpu`. |
| `gui` | `fastapi`, `uvicorn`, `ruamel.yaml` | `bioaccx gui`, a GUI no navegador para configurar, executar e analisar |
| `umap` | `umap-learn`, `matplotlib` | A projeção UMAP e os gráficos do `bioaccx embeddings` (`umap.enabled: true`) |
| `ssh` | `paramiko` | `data_dir` remoto por SSH/SFTP (`ssh_host`) |
| `arbimon` | — | Fontes do Arbimon. O SDK `rfcx` não é publicado no PyPI, e o wheel da release deve ser instalado manualmente: `pip install https://github.com/rfcx/rfcx-sdk-python/releases/download/0.3.1/rfcx-0.3.1-py3-none-any.whl` |

TensorFlow, scikit-learn, ONNX, Hugging Face Hub e Kaggle Hub são dependências
**principais**: são sempre instaladas e não requerem extra.

`cpu` e `gpu` são declarados como mutuamente exclusivos, mas isso só é verificado
ao sincronizar este projeto; instalar `bioaccx[cpu,gpu]` como ferramenta ou como
dependência instala os dois runtimes. Apenas um deles deve ser solicitado.

## Espaço em disco

O ambiente é grande: apenas o TensorFlow ocupa cerca de 1,3 GB, e um clone
sincronizado com `cpu`, `gui`, `umap` e o grupo de desenvolvimento ocupa cerca de
2,2 GB. Uma instalação com `uv tool install` mantém a própria cópia separada.

## Aceleração por GPU (ONNX Runtime + CUDA)

Para executar o backbone ONNX na GPU, defina `onnx_providers` na configuração:

```yaml
foundation_model:
  onnx_providers: [CUDAExecutionProvider, CPUExecutionProvider]
```

O provider CUDA do ONNX Runtime requer várias bibliotecas da NVIDIA que **não** acompanham o pacote `onnxruntime-gpu` e devem ser instaladas separadamente. Com acesso de administrador (sudo), os pacotes `apt` são a opção mais simples:

```bash
sudo apt install libcurand-12 libcufft-12 libcudart-12
```

Em um ambiente virtual sem acesso ao sistema, instale os equivalentes do pip no mesmo ambiente do bioaccx:

```bash
uv pip install nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cuda-runtime-cu12
# ou: pip install nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cuda-runtime-cu12
```

Esses pacotes instalam os arquivos `.so` em `site-packages/nvidia/*/lib/`, mas o ONNX Runtime os carrega com `dlopen` antes da execução do mecanismo de importação do Python, de modo que, por padrão, eles não ficam visíveis para o linker dinâmico. A solução é um arquivo `sitecustomize.py` que os pré-carrega na inicialização do interpretador. Crie-o em:

```
<venv>/lib/python3.x/site-packages/sitecustomize.py
```

com o seguinte conteúdo:

```python
import ctypes, pathlib

_nvidia_base = pathlib.Path(__file__).parent / "nvidia"
for lib in _nvidia_base.glob("*/lib/lib*.so.*"):
    try:
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
```

Isso carrega todos os arquivos `.so` da NVIDIA no processo com `RTLD_GLOBAL`, de modo que as chamadas `dlopen` subsequentes do ONNX Runtime possam resolvê-los. Nenhuma alteração em `LD_LIBRARY_PATH` ou no ambiente do shell é necessária.

Para verificar se o provider CUDA está ativo após a configuração:

```python
import onnxruntime as ort
print(ort.get_available_providers())
# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```
