<!-- translated-from: docs/en/installation.md@6567d2912404 -->
# Instalação

## O instalador (sem precisar de Python)

No Windows — abra o PowerShell e cole:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://github.com/biodiversica/bioaccx/releases/latest/download/install.ps1 | iex"
```

No macOS (Apple Silicon) e no Linux — abra um terminal e cole:

```sh
curl -LsSf https://github.com/biodiversica/bioaccx/releases/latest/download/install.sh | sh
```

O que ele faz, em ordem:

1. Instala o [uv](https://docs.astral.sh/uv/) em `~/.local/bin`, se ainda não
   estiver lá. O uv baixa o próprio Python, então não é preciso ter um antes, e
   um Python que você já tenha fica intocado.
2. Roda `uv tool install --python 3.12 "bioaccx[cpu,gui,umap]"` — o runtime de
   CPU, a GUI no navegador e os gráficos UMAP — e coloca o `bioaccx` no seu PATH.
3. Cria uma pasta de trabalho, `~/bioaccx`, onde a GUI começa: os configs que
   você salva e os modelos que treina ficam lá, a menos que você escolha outro
   lugar.
4. Adiciona um atalho **bioaccx** que roda `bioaccx gui` nessa pasta: na Área de
   Trabalho e no menu Iniciar no Windows, `bioaccx.command` na Mesa do macOS e
   uma entrada no menu de aplicativos no Linux. Ele abre uma janela de terminal
   ao lado do navegador — essa janela é o servidor, e fechá-la (ou Ctrl+C) o
   encerra.

Rodar o instalador de novo atualiza o bioaccx. Duas variáveis de ambiente mudam
os padrões: `BIOACCX_PACKAGE` (o que instalar, ex.: `bioaccx[gpu,gui]`) e
`BIOACCX_WORKDIR` (a pasta de trabalho).

Para remover, rode `uv tool uninstall bioaccx` e apague o atalho; a pasta de
trabalho é sua, para manter ou apagar.

## Sistemas suportados

| Sistema | |
|---|---|
| Linux x86-64 e ARM64 | ✓ |
| Windows x86-64 | ✓ |
| macOS em Apple Silicon (M1 em diante) | ✓ |
| macOS em Intel | ✗ — o TensorFlow não publica versão para ele |

Python 3.12 ou 3.13, que é para o que o TensorFlow publica hoje. O instalador e
os comandos abaixo fixam o 3.12, para que um Python mais novo na máquina nunca
seja escolhido por acidente.

## Como ferramenta de linha de comando

A forma usual de usar o bioaccx se você já está à vontade num terminal.
O bioaccx executa os modelos pelo ONNX Runtime, e **não há um padrão** — escolha
`cpu` ou `gpu` na instalação. Um `bioaccx` sem extras instala sem reclamar e
falha no momento em que tenta rodar um modelo, então na prática o extra não é
opcional.

O `uv tool install` cria um ambiente isolado e coloca o `bioaccx` no seu PATH:

```bash
uv tool install --python 3.12 "bioaccx[cpu]"            # inferência na CPU
uv tool install --python 3.12 "bioaccx[gpu]"            # CUDA 12 (veja a seção de GPU abaixo)
uv tool install --python 3.12 "bioaccx[cpu,gui,umap]"   # com a GUI no navegador e os gráficos UMAP
```

O `uv tool install` **não tem a opção `--extra`** — os extras vão dentro dos
colchetes, e o argumento inteiro precisa de aspas para o shell não tentar
expandi-los.

```bash
uv tool install --python 3.12 "/caminho/do/bioaccx[cpu]"   # a partir de um clone local
uv tool upgrade bioaccx                                    # depois
uv tool uninstall bioaccx
```

## Em um projeto

```bash
uv add "bioaccx[cpu]"                       # como dependência
pip install "bioaccx[cpu]"                  # ou com o pip
```

## Trabalhando no próprio bioaccx

```bash
git clone git@github.com:biodiversica/bioaccx.git
cd bioaccx
uv sync --extra cpu                         # ou --extra gpu
uv sync --extra cpu --extra gui --extra umap
uv run bioaccx --help
uv run pytest
```

O `uv sync --extra X` **substitui** o conjunto de extras ativo em vez de somar a
ele, então nomeie todos os extras que você quer a cada execução — sincronizar só
com `--extra gui` remove o `onnxruntime` e o `umap-learn` de novo.

### Publicando uma versão

Ajuste a nova versão no `pyproject.toml` e no `bioaccx/__init__.py`, faça o
commit, e então crie e envie a tag:

```bash
git tag v0.2.0 && git push origin v0.2.0
```

A tag dispara o `.github/workflows/release.yml`: os testes e os instaladores
rodam no Linux, no macOS e no Windows, o wheel vai para o PyPI, e uma release
**rascunho** é aberta no GitHub com os instaladores anexados e as notas de
`.github/release-notes.md`. Escreva os destaques e clique em Publish — os
comandos de instalação baixam da última release *publicada*.

## Extras

| Extra | Adiciona | Quando é necessário |
|---|---|---|
| `cpu` | `onnxruntime` | Rodar qualquer modelo na CPU. Escolha este ou o `gpu`. |
| `gpu` | `onnxruntime-gpu` | Rodar com CUDA 12. Escolha este ou o `cpu`. |
| `gui` | `fastapi`, `uvicorn`, `ruamel.yaml` | `bioaccx gui` — o editor de configuração no navegador e o explorador de resultados |
| `umap` | `umap-learn`, `matplotlib` | A projeção UMAP e os gráficos do `bioaccx embeddings` (`umap.enabled: true`) |
| `ssh` | `paramiko` | `data_dir` remoto por SSH/SFTP (já é dependência principal; o extra é mantido para configs antigas) |
| `arbimon` | — | Fontes do Arbimon. O SDK `rfcx` não está no PyPI; instale o wheel da release à mão: `pip install https://github.com/rfcx/rfcx-sdk-python/releases/download/0.3.1/rfcx-0.3.1-py3-none-any.whl` |

TensorFlow, scikit-learn, ONNX, HuggingFace Hub e Kaggle Hub são dependências
**principais** — sempre instaladas, sem precisar de extra.

`cpu` e `gpu` são declarados como mutuamente exclusivos, mas isso só vale ao
sincronizar este projeto; instalar `bioaccx[cpu,gpu]` como ferramenta ou como
dependência vai lhe dar os dois runtimes sem reclamar. Peça só um.

## Espaço em disco

O ambiente é grande: só o TensorFlow tem cerca de 1,3 GB, e um clone
sincronizado com `cpu`, `gui`, `umap` e o grupo de desenvolvimento chega a uns
2,2 GB. Um `uv tool install` mantém a própria cópia de tudo isso.

## Aceleração por GPU (ONNX Runtime + CUDA)

Para rodar o backbone ONNX na GPU, defina `onnx_providers` na configuração:

```yaml
foundation_model:
  onnx_providers: [CUDAExecutionProvider, CPUExecutionProvider]
```

O provider CUDA do ONNX Runtime precisa de várias bibliotecas da NVIDIA que **não** vêm junto com o pacote `onnxruntime-gpu` e precisam ser instaladas à parte.  Os pacotes `apt` são o caminho mais simples se você tiver sudo:

```bash
sudo apt install libcurand-12 libcufft-12 libcudart-12
```

Se você estiver em um virtualenv sem acesso ao sistema, instale os equivalentes do pip no mesmo ambiente do bioaccx:

```bash
uv pip install nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cuda-runtime-cu12
# ou: pip install nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cuda-runtime-cu12
```

Esses pacotes instalam os arquivos `.so` em `site-packages/nvidia/*/lib/`, mas o ONNX Runtime os carrega via `dlopen` antes de o mecanismo de importação do Python rodar, então eles ficam invisíveis para o linker dinâmico por padrão.  A solução é um arquivo `sitecustomize.py` que os pré-carrega na inicialização do interpretador.  Crie-o em:

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

Isso carrega todos os `.so` da nvidia no processo com `RTLD_GLOBAL`, de modo que as chamadas `dlopen` seguintes do ONNX Runtime consigam resolvê-los.  Nenhuma mudança em `LD_LIBRARY_PATH` ou no ambiente do shell é necessária.

Para verificar se o provider CUDA está ativo depois da configuração:

```python
import onnxruntime as ort
print(ort.get_available_providers())
# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```
