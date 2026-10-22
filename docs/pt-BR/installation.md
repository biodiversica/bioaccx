<!-- translated-from: docs/en/installation.md@f61136fb68d7 -->
# Instalação

## O instalador (sem necessidade de Python)

No Windows, abra o PowerShell e execute:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://github.com/biodiversica/bioaccx/releases/latest/download/install.ps1 | iex"
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
4. Adiciona um atalho **bioaccx** que executa `bioaccx gui` nessa pasta: na Área
   de Trabalho e no menu Iniciar, no Windows; `bioaccx.command` na Mesa, no
   macOS; e uma entrada no menu de aplicativos, no Linux. O atalho abre uma
   janela de terminal ao lado do navegador. Essa janela executa o servidor, e
   fechá-la (ou pressionar Ctrl+C) encerra o servidor.

Executar o instalador novamente atualiza o bioaccx. Duas variáveis de ambiente
alteram os padrões: `BIOACCX_PACKAGE` (o pacote a instalar, por exemplo
`bioaccx[gpu,gui]`) e `BIOACCX_WORKDIR` (a pasta de trabalho).

Para desinstalar o bioaccx, execute `uv tool uninstall bioaccx` e exclua o
atalho. A pasta de trabalho não é removida e pode ser mantida ou excluída.

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
| `gui` | `fastapi`, `uvicorn`, `ruamel.yaml` | `bioaccx gui`, o editor de configuração no navegador e o explorador de resultados |
| `umap` | `umap-learn`, `matplotlib` | A projeção UMAP e os gráficos do `bioaccx embeddings` (`umap.enabled: true`) |
| `ssh` | `paramiko` | `data_dir` remoto por SSH/SFTP (já é uma dependência principal; o extra é mantido por compatibilidade com instalações antigas) |
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
