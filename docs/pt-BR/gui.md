<!-- translated-from: docs/en/gui.md@d8ea8db08849 -->
# A GUI no navegador

O `bioaccx gui` abre um editor de configuração no navegador. Ele grava os mesmos
arquivos YAML lidos por todos os outros comandos, e as execuções continuam
podendo ser iniciadas pelo terminal.

```bash
uv tool install "bioaccx[cpu,gui]"      # a GUI requer o extra [gui]

bioaccx gui                             # começa a partir de um modelo de configuração
bioaccx gui my_config.yaml              # abre uma configuração existente
bioaccx gui --port 9000 --no-open       # define a porta, sem abrir o navegador
bioaccx gui --lang pt-BR                # abre a interface em português
```

O formulário é **gerado a partir das dataclasses de configuração** e, portanto,
oferece sempre exatamente as keys suportadas por esta versão do bioaccx, com
os valores padrão reais e a documentação escrita ao lado delas no código-fonte.
Não existe uma segunda cópia do schema que possa ficar desatualizada.

Em comparação com um editor de texto, a GUI oferece o seguinte:

- **O seletor de registro.** Ao escolher um backbone, são exibidos os valores
  que ele utiliza (taxa de amostragem, janela, tamanho do embedding e nomes dos
  tensores), lidos do mesmo registro impresso pelo `bioaccx registry` e marcados
  com `Modelo base: 0xbb10`. Abaixo deles, uma segunda linha indica a origem dos
  pesos (`huggingface · biodiversica/BirdNET-onnx-backbone · model_backbone.onnx`,
  o identificador do Kaggle ou um caminho local), de modo que o download
  implícito em um ID fique visível já na lista, e não apenas nos campos
  esmaecidos `hf_repo` e `hf_filename` mais abaixo. Esses valores são exibidos,
  e não gravados: o arquivo mantém sua única linha `registry_id:`, o carregador
  incorpora os demais valores na leitura, e somente um valor digitado em um
  campo se torna uma sobrescrita efetiva.
- **Apenas as configurações aplicáveis.** As seções de Keras e scikit-learn são
  exibidas conforme `training.classifier`, de modo que as configurações de um
  head que não será treinado ficam ocultas. Alternar entre elas preserva no
  arquivo as configurações do outro head.
- **Validação antes da execução.** O rascunho é verificado com a mesma função
  `load_config` usada pela CLI, de modo que um `registry_id` inválido, um bloco
  de augmentation sem fonte de ruído ou um softmax agrupado sem `label_groups` é
  detectado durante a edição, e não pouco depois do início de uma execução
  longa.
- **Facilidade de descoberta.** Cerca de 120 keys de configuração, divididas
  entre configurações comuns e avançadas, são apresentadas com seus textos de
  ajuda, sem necessidade de consultar a
  [referência de configuração](config-reference.md).
- **Os comentários são preservados.** Os arquivos são lidos e gravados por um
  processo de YAML que preserva os comentários: abrir o `example_config.yaml` e
  salvá-lo novamente não altera nenhum byte, e editar dois campos altera apenas
  esses dois campos.
- **Um layout compacto.** As cinco seções aparecem inicialmente recolhidas,
  apresentando um índice curto em vez de 120 campos, e são expandidas com um
  clique no título; esse estado é lembrado por navegador. O botão à direita da
  barra alterna o tema: ◐ segue a configuração do sistema, ☀ claro, ☾ escuro.
- **Inglês ou português.** O seletor ao lado do botão de tema altera o idioma da
  interface; `--lang pt-BR` define o idioma de um navegador em que nenhum idioma
  foi escolhido. Nomes dos campos, textos de ajuda, seções, botões e mensagens
  são todos traduzidos, e posicionar o cursor sobre o nome de um campo mostra a
  key YAML que ele grava. Consulte
  [Idiomas da interface](#idiomas-da-interface).

## Explorando o que uma execução produziu

A aba **Análise** lê o diretório `custom_models/` (ou o `--models-dir`) e
apresenta os arquivos que toda execução já grava em uma forma que pode ser
explorada. Nada é recalculado, de modo que a aba também funciona com modelos
treinados antes da existência deste recurso.

- **Lista de modelos:** cada execução é exibida como um card com o backbone, o
  número de classes, o F1 macro extraído da tabela de avaliação (e o *F1 dos
  incluídos*, o F1 macro sem as classes de `exclude_labels`, quando a execução
  excluiu alguma), a indicação de que há ou não um mapa de embeddings e, para
  uma execução treinada com mixup de áudio, o número de misturas sintéticas
  usadas no treino. O diretório pode ser editado no topo da lista, de modo que
  os resultados podem ser lidos de qualquer local sem reiniciar; o
  `--models-dir` define apenas o diretório inicial.
- **Métricas:** a tabela de avaliação por classe, que pode ser ordenada. Clicar
  em *F1* traz as classes mais fracas para o topo, e valores abaixo de 0,5 são
  destacados. As classes de `exclude_labels` aparecem esmaecidas e ficam fora da
  linha *Média macro (rótulos incluídos)*. Para execuções treinadas antes da
  existência dessa linha, ela é calculada a partir das linhas por classe, e a
  *Comparação* também a inclui. Para uma execução treinada com mixup de áudio,
  cada classe mostra também suas janelas reais de treino, as misturas cujo alvo
  a contém (posicionar o cursor mostra com quais classes ela foi misturada) e a
  proporção em misturas, isto é, a parte sintética do seu sinal de treino, ao
  lado das métricas, que são sempre calculadas sobre janelas reais de teste. O
  conjunto de teste de misturas, quando gerado, tem uma tabela própria logo
  abaixo.
- **Mapa:** a projeção UMAP em um canvas, colorida por rótulo ou por cluster do
  KMeans e filtrável por divisão. As cores são geradas para o número de classes
  efetivamente presentes, de modo que duas classes nunca compartilham a mesma
  cor, e são as mesmas da figura `_umap.png`, permitindo comparar diretamente o
  mapa e a figura. **Clicar em um ponto reproduz o trecho a partir do qual ele
  foi calculado**, com o espectrograma ao lado. Isso não é possível pela CLI e
  mostra *quais gravações* se encontram em uma região em que as classes se
  confundem.
- **Comparação:** duas execuções lado a lado, com as diferenças de F1 por classe
  e exatamente as configurações que diferem entre elas, lidas dos metadados de
  cada execução. Mostra o efeito de uma alteração, como um valor diferente de
  `hidden_units`, sem a necessidade de comparar dois relatórios manualmente.
  Quando alguma das execuções usou mixup de áudio, a proporção em misturas de
  cada classe nas duas execuções aparece ao lado da diferença de F1, e as
  configurações do mixup são incluídas na comparação.

Para reproduzir um ponto, a GUI precisa identificar a sua amostra. Por isso, as
execuções gravam uma coluna `key` no `<model>_umap.csv`. Projeções gravadas
antes da existência dessa coluna continuam suportadas: as keys são inferidas
pela ordem das linhas, mas apenas quando a projeção e a lista do dataset têm o
mesmo tamanho *e* concordam rótulo a rótulo. Caso contrário, o mapa informa que
os pontos não podem ser associados ao seu áudio, em vez de fazer uma
suposição. O mapa indica qual dos dois métodos foi usado.

Um ponto reproduz a janela que o modelo transformou em embedding. Quando a
execução exportou o dataset (`export_dataset`), trata-se do arquivo WAV
exportado em `<model>/dataset/`. Caso contrário, a janela é extraída da gravação
de origem registrada no `<model>_dataset_list.csv` e reconstruída como foi
usada: uma cópia com ruído é misturada ao seu arquivo de ruído no SNR
correspondente, e um sinal curto é posicionado no seu deslocamento aleatório.
Listas gravadas antes do registro do caminho do ruído contêm apenas o nome do
arquivo de ruído, que é então procurado no `augmentation_dir` da execução. Essas
listas também não registram a duração do sinal, de modo que suas janelas
deslocadas são reproduzidas sem o deslocamento. Se o áudio tiver sido movido ou
estiver em outra máquina, o ponto ainda mostra seus metadados e informa que o
arquivo está ausente, em vez de falhar.

## Idiomas da interface

O editor está disponível em inglês e em português do Brasil. O seletor na barra
altera o idioma sem recarregar a página e lembra a escolha naquele navegador; o
`bioaccx gui --lang pt-BR` define o idioma de um navegador em que nenhum idioma
foi escolhido.

```bash
bioaccx gui --lang pt-BR        # abre em português
bioaccx gui -l pt               # qualquer tag que possa ser resolvida; tags desconhecidas geram um aviso e usam en
```

Adicionar um idioma requer um único arquivo: colocar `<código>.json` em
`bioaccx/gui/locales/`, ao lado do `en.json`, o inclui no seletor. Um arquivo de
idioma precisa conter apenas os textos que traduz. O servidor o mescla sobre os
textos em inglês, de modo que uma tradução incompleta mostra o restante em
inglês, em vez de keys sem tradução.

Dois aspectos são intencionais:

- **A key é sempre exibida ao posicionar o cursor.** Os nomes dos campos são
  escritos por extenso em todos os idiomas (`hf_repo` aparece como *Hugging Face
  repository* em inglês e como *Repositório do Hugging Face* em português), de
  modo que todo nome de campo exibe seu caminho de configuração como dica, e o
  formulário continua identificando o que grava no arquivo.
- **Os textos em inglês são definidos no código-fonte, e não no `en.json`.** Os
  textos de ajuda dos campos vêm das dataclasses de configuração, e os nomes dos
  campos em inglês são derivados das keys, de modo que o `en.json` contém
  apenas os textos criados pela própria interface do navegador. Uma key nova
  em `config.py` aparece, portanto, em todos os idiomas assim que é adicionada:
  em inglês até ser traduzida, e nunca vazia nem como key ausente.

As execuções não são traduzidas: o log transmitido para a página é a saída da
própria CLI, idêntica ao que o terminal mostraria.

## Iniciando uma execução pelo editor

A aba **Execução** executa a configuração indicada no campo de caminho. Depois
de selecionar `train`, `dataset`, `embeddings` ou `validate` e pressionar
**Executar**, a saída é transmitida para a página à medida que é produzida, com
uma barra de progresso guiada pelos marcadores `[3/5]` que o pipeline já
imprime. **Cancelar** envia a mesma interrupção que o Ctrl+C, de modo que a
execução é encerrada da forma habitual. Enquanto uma execução está em andamento,
um ponto na aba **Execução** indica isso nas demais abas.

Três aspectos são intencionais:

- **A execução lê o arquivo em disco, e não o rascunho no navegador.** Uma
  execução com alterações não salvas é recusada com uma mensagem, em vez de ser
  iniciada, de modo que toda execução pode sempre ser reproduzida a partir de um
  terminal.
- **O comando equivalente é exibido ao lado do botão**, com um botão de cópia.
  Tudo o que o editor faz também está disponível no shell.
- **A execução é um processo separado, em sua própria sessão.** Ela não
  compartilha memória com o editor, uma falha não interrompe o editor, e fechar
  o navegador, ou reiniciar o servidor, não interrompe o treinamento. Reabrir a
  página reconecta a uma execução já em andamento.

Apenas esses quatro comandos podem ser iniciados. Os comandos de cirurgia de
modelo (`merge`, `extract-head`, `convert-head`) recebem caminhos de modelos em
vez de um arquivo de configuração e, por isso, permanecem no terminal. É
permitida apenas uma execução por vez: uma segunda execução é recusada, e não
enfileirada, porque duas execuções na mesma máquina disputam a mesma CPU, a
mesma GPU e o mesmo diretório de saída.

### Augmentation por fonte

Cada fonte tem um de três estados: **herdar** o bloco compartilhado, **bloco
próprio** ou **sem augmentation**. Escolher *bloco próprio* exibe todo o
conjunto de configurações de augmentation apenas para aquela fonte. O bloco
próprio de uma fonte substitui integralmente o bloco herdado:

```yaml
dataset:
  label_mode: subfolders
  augmentation:                     # compartilhado pelas fontes que não definem o próprio
    augmentation_dir: /noise/general
    snr_levels: [20, 10]

  sources:
    - data_dir: /audio/quiet_site
      augmentation:                 # bloco próprio, que substitui o compartilhado
        augmentation_dir: /noise/rain
        snr_levels: [3]

    - data_dir: /audio/clean_recordings
      augmentation: null            # sem augmentation

    - data_dir: /audio/normal_site  # sem configuração: herda o bloco compartilhado
```

Alternar uma fonte entre os três estados preserva o seu bloco próprio, de modo
que mudar para *sem augmentation* e voltar não descarta as configurações. Campos
de nível da execução (`test_ratio`, `random_seed`, `audio_extensions` e
credenciais) nunca podem ser sobrescritos por fonte e aparecem marcados como
**Nível da execução** no formulário.

O arquivo que está sendo gravado é exibido ao lado do formulário durante a
edição, de modo que o conteúdo exibido é exatamente o que é gravado em disco.
Marcar **Editar o YAML diretamente** permite digitar YAML nesse painel, de modo
que configurações não cobertas pelo formulário podem ser escritas manualmente.

A prévia fica oculta por padrão e é exibida ou ocultada com **Mostrar arquivo**
/ **Ocultar** no seu cabeçalho. O resumo da validação permanece visível em ambos
os casos, e a escolha é lembrada.

Segredos (`xc_api_key`) nunca são enviados de volta ao navegador: uma chave
existente aparece como definida, e deixar o campo em branco a mantém inalterada
no arquivo.

Por padrão, o servidor escuta em `127.0.0.1` e só pode ser acessado a partir da
máquina local. Escutar em qualquer outro endereço gera um token de acesso, que é
incluído no link impresso e depois armazenado como cookie; `--no-auth` o
desativa.

Para acessar o editor a partir de outra máquina, é preferível escutar apenas na
interface necessária, em vez de em todas:

```bash
bioaccx gui --host "$(tailscale ip -4)"   # apenas a tailnet, não a rede local
bioaccx gui --host 192.168.0.30           # apenas esta rede local
bioaccx gui --host 0.0.0.0                # todas as interfaces; imprime cada URL
```

Pela Tailscale, o tráfego já é criptografado de ponta a ponta com WireGuard, de
modo que o token pode trafegar com segurança sobre HTTP simples dentro da
tailnet. O editor não deve ser exposto com o `tailscale funnel`, que o publica
na internet pública, porque o editor pode ler e gravar arquivos em qualquer
local acessível ao usuário que o executa.
