<!-- translated-from: docs/en/gui.md@5d9e7ee8e7fe -->
# A GUI no navegador

O `bioaccx gui` abre um editor de configuração no navegador. Ele escreve os
mesmos arquivos YAML que todos os outros comandos leem — o terminal continua
sendo o lugar de onde as execuções partem.

```bash
uv tool install "bioaccx[cpu,gui]"      # a GUI precisa do extra [gui]

bioaccx gui                             # começa de um modelo em branco
bioaccx gui my_config.yaml              # abre uma configuração existente
bioaccx gui --port 9000 --no-open       # escolhe a porta, sem abrir o navegador
bioaccx gui --lang pt-BR                # abre em português
```

O formulário é **gerado a partir das dataclasses de configuração**, então ele
sempre oferece exatamente as keys que esta versão do bioaccx entende, com os
valores padrão de verdade e a documentação escrita ao lado delas no código. Não
existe uma segunda cópia do schema para ficar desatualizada.

O que ele dá além de um editor de texto:

- **O seletor de registro.** Escolher um backbone mostra o que ele realmente
  usa — taxa de amostragem, janela, tamanho do embedding, nomes dos tensores —
  lido do mesmo registro que o `bioaccx registry` imprime, e marcado com
  `from 0xbb10`. Abaixo disso, uma segunda linha diz de onde vêm os pesos —
  `huggingface · biodiversica/BirdNET-onnx-backbone · model_backbone.onnx`, ou
  o handle do Kaggle, ou um caminho local — de modo que o download implícito no
  ID fique visível já no menu, e não só nos campos esmaecidos `hf_repo` /
  `hf_filename` mais abaixo. Os valores são mostrados, não escritos: o arquivo
  mantém sua única linha `registry_id:`, o carregador junta o resto na leitura,
  e digitar por cima de um campo é o que o torna uma sobrescrita de verdade.
- **Só o que se aplica.** Os blocos de Keras e sklearn aparecem conforme
  `training.classifier`, então você não rola ajustes de um head que não será
  treinado. Alternar entre eles preserva no arquivo os ajustes do outro.
- **Validação antes da execução.** O rascunho é conferido com o mesmo
  `load_config` que a CLI usa, então um `registry_id` inválido, um bloco de
  augmentation sem fonte de ruído ou um softmax agrupado sem `label_groups` é
  pego enquanto você edita, e não trinta segundos depois do início de três horas
  de treinamento.
- **Descobribilidade.** Cerca de 120 keys de configuração, separadas entre
  comuns e avançadas, cada uma com seu texto de ajuda — em vez de percorrer a
  [referência de configuração](config-reference.md).
- **Seus comentários sobrevivem.** Os arquivos são lidos e escritos por um
  round-trip de YAML que preserva comentários: abrir o `example_config.yaml` e
  salvá-lo de volta não muda um byte, e editar dois campos muda apenas esses
  dois campos.
- **Ele sai da frente.** As cinco seções abrem recolhidas — um índice curto em
  vez de 120 campos — e se expandem com um clique no título, lembrado por
  navegador. O botão à direita da barra alterna o tema: ◐ segue o sistema,
  ☀ claro, ☾ escuro.
- **Inglês ou português.** O seletor ao lado do botão de tema troca a interface;
  `--lang pt-BR` decide o que um navegador que nunca escolheu vê. Rótulos dos
  campos, textos de ajuda, seções, botões e mensagens são todos traduzidos — e
  como um rótulo traduzido deixa de soletrar a key YAML, passar o mouse sobre
  ele mostra a key que será escrita. Veja
  [Idiomas da interface](#idiomas-da-interface).

## Explorando o que uma execução produziu

A aba **análise** lê o diretório `custom_models/` (ou o `--models-dir`) e
transforma os arquivos que toda execução já escreve em algo que se pode
interrogar. Nada é recalculado, então funciona com modelos treinados muito antes
de isso existir.

- **Lista de modelos** — cada execução como um card: backbone, número de
  classes, F1 macro tirado da tabela de avaliação (mais o *F1 incl.*, o F1
  macro sem as classes de `exclude_labels`, quando a execução excluiu alguma),
  se há ou não um mapa de embeddings e — para uma execução treinada com mixup de
  áudio — com quantas misturas sintéticas ela treinou. O diretório é editável no topo da lista, então dá para ler
  resultados de qualquer lugar sem reiniciar; o `--models-dir` só define onde
  ela começa.
- **Métricas** — a tabela de avaliação por classe, ordenável. Clique em *F1*
  para trazer as classes mais fracas ao topo; qualquer valor abaixo de 0,5 é
  marcado. As classes de `exclude_labels` aparecem esmaecidas e ficam fora da
  linha `OVERALL (Macro-avg, included)`. Para execuções treinadas antes de essa
  linha existir, ela é calculada a partir das linhas por classe; a *Comparação*
  também a mostra. Uma execução treinada com mixup de áudio acrescenta, por
  classe, suas janelas reais de treino, as misturas cujo alvo a contém (passe o
  mouse para ver com o que foi misturada) e a parcela mista — a parte sintética
  do seu sinal de treino — ao lado das métricas, que são sempre das janelas
  reais de teste. O conjunto de teste misto, quando gerado, tem a sua própria
  tabela logo abaixo.
- **Mapa** — a projeção UMAP em um canvas, colorida por rótulo ou por cluster do
  KMeans e filtrável por split. As cores são geradas para o número de classes
  realmente presentes — nunca duas classes com a mesma cor — e são as mesmas da
  figura `_umap.png`, de modo que o mapa e a figura se leiam do mesmo jeito.
  **Clique em um ponto para ouvir o trecho de onde ele veio**, com o
  espectrograma ao lado. É isso que a CLI não faz: ver *quais gravações* estão
  numa região confusa.
- **Comparar** — duas execuções lado a lado, com as diferenças de F1 por classe
  e exatamente quais ajustes diferem entre elas, lidos dos metadados de cada
  uma. Responde "o que mudar o `hidden_units` fez de fato" sem comparar dois
  relatórios a olho. Quando alguma das execuções usou mixup de áudio, a parcela
  mista de cada classe dos dois lados aparece ao lado da diferença de F1, e os
  ajustes do mixup entram na comparação.

Clicar num ponto exige saber de qual amostra ele é. As execuções agora escrevem
uma coluna `key` no `<model>_umap.csv` justamente para isso. Projeções escritas
antes dessa coluna existir continuam funcionando: as keys são inferidas pela
ordem das linhas, mas só quando a projeção e a lista do dataset têm o mesmo
tamanho *e* concordam rótulo a rótulo — caso contrário o mapa diz que os pontos
não podem ser rastreados, em vez de chutar. O mapa informa qual dos dois casos
usou.

O áudio de origem é lido dos caminhos registrados no
`<model>_dataset_list.csv`. Se um dataset mudou de lugar ou está em outra
máquina, o ponto ainda mostra seus metadados e avisa que o arquivo está
faltando, em vez de falhar.

## Idiomas da interface

O editor vem em inglês e português do Brasil. O seletor na barra troca o idioma
sem recarregar a página e lembra a escolha naquele navegador; o
`bioaccx gui --lang pt-BR` define o que um navegador que nunca escolheu vê.

```bash
bioaccx gui --lang pt-BR        # abre em português
bioaccx gui -l pt               # qualquer tag que resolva; as desconhecidas avisam e usam en
```

Adicionar um idioma é um arquivo só: coloque `<código>.json` em
`bioaccx/gui/locales/`, ao lado do `en.json`, e ele aparece no seletor. Um
locale só precisa carregar o que traduz — o servidor o mescla sobre o inglês,
então uma tradução pela metade mostra inglês no resto, em vez de exibir keys
cruas.

Duas coisas são deliberadas:

- **A key está sempre a um passar de mouse.** Em inglês o rótulo de um campo
  *é* a sua key por extenso (`sample_rate` se lê como *sample rate*);
  traduzido, não é — então todo rótulo carrega seu caminho de configuração como
  tooltip, e o formulário continua dizendo o que escreve no arquivo.
- **O inglês vive no código, não no `en.json`.** Os rótulos e textos de ajuda
  vêm das dataclasses de configuração, então o `en.json` guarda apenas os textos
  que o próprio navegador inventa. Uma key nova em `config.py` portanto
  aparece em todos os idiomas no dia em que é adicionada — em inglês até alguém
  traduzir, nunca em branco nem como key faltando.

As execuções não são traduzidas: o log transmitido para a página é a saída da
própria CLI, byte a byte o que o terminal mostraria.

## Iniciando uma execução pelo editor

A aba **executar** roda a configuração indicada na caixa de caminho: escolha `train`,
`dataset`, `embeddings` ou `validate`, aperte **Run**, e a saída chega na página
conforme acontece, com uma barra de progresso guiada pelos marcadores `[3/5]`
que o pipeline já imprime. **Cancel** envia a mesma interrupção que o Ctrl+C
enviaria, então a execução para do jeito de sempre. Enquanto uma execução
está em andamento, um ponto na aba **executar** avisa disso nas outras abas.

Três coisas são deliberadas:

- **A execução lê o arquivo em disco, não o rascunho no navegador.** Alterações
  não salvas são recusadas com uma mensagem em vez de rodarem silenciosamente,
  então o que rodou sempre pode ser reproduzido a partir de um terminal.
- **O comando equivalente é mostrado ao lado do botão** (com um botão de
  copiar). Nada do que o editor faz é indisponível no shell.
- **A execução é um processo separado, na própria sessão.** Ele não compartilha
  memória com o editor, uma queda dele não derruba o editor, e fechar o
  navegador — ou reiniciar o servidor — não interrompe o treinamento. Reabrir a
  página reconecta a uma execução em andamento.

Só esses quatro comandos podem ser iniciados. Os comandos de cirurgia de modelo
(`merge`, `extract-head`, `convert-head`) recebem caminhos de modelos em vez de
uma configuração, então ficam no terminal. Uma execução por vez: uma segunda é
recusada em vez de enfileirada, porque duas execuções na mesma máquina disputam
a mesma CPU, GPU e diretório de saída.

### Augmentation por fonte

Cada fonte escolhe um de três estados — **herdar** o bloco compartilhado,
**bloco próprio** ou **sem augmentation** — e escolher *bloco próprio* abre todo
o conjunto de ajustes de augmentation só para aquela fonte. O bloco próprio de
uma fonte substitui inteiramente o herdado:

```yaml
dataset:
  label_mode: subfolders
  augmentation:                     # compartilhado pelas fontes que não dizem nada
    augmentation_dir: /noise/general
    snr_levels: [20, 10]

  sources:
    - data_dir: /audio/quiet_site
      augmentation:                 # bloco próprio, substitui o compartilhado
        augmentation_dir: /noise/rain
        snr_levels: [3]

    - data_dir: /audio/clean_recordings
      augmentation: null            # abre mão por completo

    - data_dir: /audio/normal_site  # não diz nada, herda o bloco compartilhado
```

Alternar uma fonte entre os três estados preserva o bloco dela, então ir para
*sem augmentation* e voltar não descarta os ajustes. Campos de nível da execução
(`test_ratio`, `random_seed`, `audio_extensions`, credenciais) nunca podem ser
sobrescritos por fonte, e aparecem marcados como `run level` no formulário.

O arquivo que está sendo escrito é mostrado ao lado do formulário enquanto você
edita, então o que você vê é exatamente o que vai para o disco. Marque **edit
directly** para digitar YAML nesse painel — o que o formulário não cobre pode
ser escrito à mão.

Essa prévia fica escondida por padrão e é alternada por **Show file** / **Hide**
no cabeçalho dela; o resumo da validação continua visível de qualquer forma, e a escolha é lembrada.

Segredos (`xc_api_key`) nunca são devolvidos ao navegador: uma chave existente
aparece como definida, e deixar o campo em branco a mantém inalterada no
arquivo.

Por padrão o servidor escuta em `127.0.0.1` e só é acessível desta máquina.
Escutar em qualquer outro endereço gera um token de acesso, incluído no link
impresso e depois guardado como cookie; `--no-auth` desativa isso.

Para alcançar o editor de outra máquina, prefira escutar naquela interface
específica em vez de em todas:

```bash
bioaccx gui --host "$(tailscale ip -4)"   # só a tailnet — não a sua LAN
bioaccx gui --host 192.168.0.30           # só esta LAN
bioaccx gui --host 0.0.0.0                # todas as interfaces; imprime cada URL
```

Pela Tailscale o tráfego já é criptografado ponta a ponta com WireGuard, então o
token trafega em segurança sobre HTTP simples dentro da tailnet. Não coloque
isto atrás do `tailscale funnel` — isso o publica na internet aberta, e o editor
pode ler e escrever arquivos em qualquer lugar que o usuário que o roda puder.
