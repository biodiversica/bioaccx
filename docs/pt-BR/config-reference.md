# Referência de configuração

Todas as keys aceitas no arquivo de configuração, com o valor padrão e o que
cada uma faz.

Esta página é **gerada** a partir de `bioaccx/config.py` — as mesmas dataclasses
que constroem o formulário do navegador — pelo `tools/gen_docs.py`. As
descrições em português vêm de `bioaccx/gui/locales/pt-BR.json`, o mesmo arquivo
que traduz a GUI: traduzir um campo lá traduz esta página junto. Um campo ainda
não traduzido aparece em inglês, nunca em branco.

A prosa e os exemplos completos estão em [Configuration](../en/configuration.md)
(em inglês).

<!-- generated:config-reference — edite bioaccx/config.py e o locale, não este bloco -->

## `foundation_model` — Modelo base

Seleção do modelo backbone e parâmetros relacionados a ele.

| Key | Padrão | Descrição |
|---|---|---|
| `registry_id` | `null` | Seleciona um backbone conhecido e carrega todos os seus padrões. Qualquer campo definido junto sobrescreve o padrão do registro. |
| `name` | *obrigatório* | Nome do modelo, usado nos relatórios e nos nomes dos arquivos de saída. |
| `version` | `unknown` | String de versão do modelo. |
| `data_type` | `FP32` | Precisão dos pesos, por exemplo FP32 ou INT8. |
| `format` | `onnx` | Formato do modelo em disco. |
| `source` | `local` | De onde o modelo é carregado: um caminho local, o Hugging Face Hub ou o Kaggle. |
| `path` | `null` | Caminho do arquivo ou diretório local do modelo. |
| `hf_repo` | `null` | Repositório do Hugging Face, por exemplo "biodiversica/birdnet". |
| `hf_filename` | `null` | Arquivo a baixar do repositório do Hugging Face. |
| `hf_revision` | `null` | Branch, tag ou commit do repositório do Hugging Face. |
| `kaggle_handle` | `null` | Identificador do modelo no Kaggle, por exemplo "google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier". |
| `kaggle_filename` | `null` | Arquivo a usar dentro do diretório baixado do Kaggle (ONNX ou TFLite). |
| `sample_rate` | `48000` | Taxa de amostragem esperada pelo backbone, em Hz. |
| `window_samples` | `null` | Comprimento da janela em amostras. Tem prioridade sobre window_seconds. |
| `window_seconds` | `null` | Comprimento da janela em segundos, por exemplo 3.0. |
| `input_name` | `input` | Nome do tensor de entrada de áudio. |
| `output_name` | `embedding` | Nome do tensor de saída do embedding. |
| `embedding_size` | `1024` | Dimensão do vetor de embedding. |
| `onnx_providers` | `null` | Execution providers do ONNX Runtime, por exemplo ["CUDAExecutionProvider", "CPUExecutionProvider"]. Quando não definido, o ONNX Runtime usa a própria ordem de prioridade. |
| `onnx_batch_size` | `1` | Número de janelas de áudio processadas em uma única chamada de inferência ONNX. Valores acima de 1 ativam o modo de lote na GPU. Aplica-se apenas quando format é "onnx". |
| `tflite_output_tensor_offset` | `0` | Deslocamento aplicado ao índice do tensor de saída TFLite para chegar ao tensor de embedding. Use -1 para modelos como o BirdNET TFLite, em que o classificador é a primeira saída e o embedding fica uma posição antes dele na lista de tensores do grafo. |
| `tflite_trim_to_embedding` | `true` | Corta o grafo TFLite até o tensor de embedding quando o modelo calcula mais do que isso, como um classificador embutido ou saídas auxiliares. Isso é feito uma vez ao carregar, e todas as janelas seguintes já pulam os ramos descartados. Use false para executar o modelo exatamente como foi distribuído. |

## `dataset` — Dataset

Seleção de fontes de áudio, tipos de rótulos, pré-processamento e criação do dataset para treinamento do modelo.

| Key | Padrão | Descrição |
|---|---|---|
| `data_dir` | `[]` | Fonte ou fontes de áudio locais. Pode ficar vazio quando apenas ext_table_file é usado. |
| `label_mode` | `subfolders` | Como os rótulos estão organizados em data_dir. |
| `table_file` | `null` | Tabela de rótulos CSV ou TSV, usada quando label_mode é "table". |
| `audio_extensions` | `["wav", "flac", "mp3", "ogg"]` | Extensões de arquivo de áudio aceitas (sem diferenciar maiúsculas de minúsculas). — **nível da execução** (não pode ser sobrescrito por fonte) |
| `overlap` | `0.0` | Sobreposição entre janelas consecutivas na divisão dos trechos rotulados, de 0.0 a 1.0. |
| `embedding_workers` | `4` | Workers paralelos da extração de embeddings (limitado aos núcleos de CPU disponíveis) — **nível da execução** (não pode ser sobrescrito por fonte) |
| `embeddings_cache_path` | `null` | Caminho onde procurar embeddings .npy pré-computados antes de executar o modelo. — **nível da execução** (não pode ser sobrescrito por fonte) |
| `append_dataset_path` | `null` | Caminho de um dataset já exportado (subpastas com train/test). Amostras novas de data_dir que ainda não estejam nele são acrescentadas. — **nível da execução** (não pode ser sobrescrito por fonte) |
| `test_ratio` | `0.2` | Fração dos dados reservada para teste. Ignorada quando o dataset já define a divisão. — **nível da execução** (não pode ser sobrescrito por fonte) |
| `random_seed` | `42` | Semente da divisão, do embaralhamento e dos deslocamentos de ruído. — **nível da execução** (não pode ser sobrescrito por fonte) |
| `filename_col` | `filename` | Coluna com o nome do arquivo de áudio. |
| `label_col` | `label` | Coluna com o rótulo da classe. |
| `start_col` | `start_time` | Coluna com o início do trecho, em segundos. |
| `end_col` | `end_time` | Coluna com o fim do trecho, em segundos. |
| `split_col` | `split` | Coluna opcional com a divisão, com os valores "train" ou "test". |
| `ext_table_file` | `null` | Tabela do iNaturalist: um arquivo CSV ou TSV com linhas de observation_id, opcionalmente misturadas com linhas de filename. |
| `obs_id_col` | `observation_id` | Coluna com o ID da observação no iNaturalist. |
| `sound_index_col` | `sound_index` | Coluna com o índice do som no iNaturalist (a partir de 0). |
| `xc_id_col` | `xc_id` | Coluna com o ID da gravação no Xeno-canto. |
| `ext_cache_dir` | `null` | Cache local do áudio remoto baixado. O padrão é ~/.cache/bioaccx/ext. — **nível da execução** (não pode ser sobrescrito por fonte) |
| `xc_api_key` | `null` | Chave da API v3 do Xeno-canto, necessária para os metadados (busca do nome científico). O áudio pode ser baixado sem chave pela URL de download direto. As chaves são emitidas em https://xeno-canto.org/explore/api. — **secreto** (nunca devolvido ao navegador) — **nível da execução** (não pode ser sobrescrito por fonte) |
| `arbimon_credentials_path` | `null` | Arquivo de credenciais do Arbimon (RFCx), gerado por rfcx.Client().authenticate(persisted_credentials_path=...). — **nível da execução** (não pode ser sobrescrito por fonte) |
| `arbimon_stream_id_col` | `stream_id` | Coluna com o ID do stream (site) no Arbimon. |
| `arbimon_date_col` | `date` | Coluna com a data local da gravação. |
| `arbimon_time_col` | `time` | Coluna com a hora local de início da gravação. |
| `arbimon_utc_offset_col` | `utc_offset` | Coluna com o fuso horário (offset UTC) da data e da hora da gravação. |
| `filter` | `null` | Filtro aplicado ao áudio antes da divisão em segmentos: "hpf" (passa-altas), "lpf" (passa-baixas), "bpf" (passa-banda) ou null. |
| `filter_freq` | `null` | Frequência de corte em Hz para hpf e lpf; [hz_baixo, hz_alto] para bpf. |
| `filter_order` | `5` | Ordem do filtro Butterworth. |
| `speed` | `1.0` | Multiplicador da velocidade de reprodução: acima de 1 é mais rápido e mais curto; abaixo de 1, mais lento e mais longo. Os tempos dos rótulos são ajustados na mesma proporção: novo_tempo = tempo_antigo / speed. |
| `ssh_host` | `null` | Servidor SSH com o áudio. Quando definido, os caminhos de data_dir são tratados como caminhos remotos neste servidor e espelhados localmente com paramiko antes da execução do pipeline. Requer o paramiko (pip install paramiko). |
| `ssh_user` | `null` | Nome do usuário SSH. O padrão é o usuário local. |
| `ssh_port` | `22` | Porta SSH. |
| `ssh_key_path` | `null` | Caminho do arquivo de chave privada. |
| `random_sample_shift` | `false` | Quando true, amostras mais curtas que a janela do modelo base são posicionadas em um deslocamento aleatório dentro da janela, em vez de começarem sempre na posição 0. Cada cópia aumentada da mesma amostra recebe um deslocamento diferente. |
| `min_anchor_fraction` | `0.1` | Fração mínima da janela que precisa ser áudio novo (ainda não coberto) para que o segmento âncora no fim do arquivo seja emitido. Isso evita segmentos quase duplicados quando os arquivos são apenas um pouco maiores que a janela (por exemplo, 3,013 s com uma janela de 3 s). Use 0.0 para sempre emitir o segmento âncora. |
| `sources` | `null` | Combina vários caminhos, cada um com suas próprias configurações, em uma única execução de criação do dataset. Cada bloco de fonte herda os campos do bloco principal e os sobrescreve com os seus. Campos marcados como "Nível da execução" nunca são sobrescritos por uma fonte. |

### `dataset.augmentation` — Augmentation

| Key | Padrão | Descrição |
|---|---|---|
| `snr_levels` | *obrigatório* | Valores de SNR em dB. É gerada uma cópia aumentada por arquivo de ruído para cada nível. |
| `augmentation_dir` | `null` | Diretório de arquivos WAV usados como fontes de ruído. Opcional quando augmentation_labels está definido; se os dois forem dados, o conjunto de ruído é a união. |
| `augmentation_labels` | `null` | Rótulos do dataset em criação cujo áudio é usado como fonte adicional de ruído (além de augmentation_dir). As amostras desses rótulos são misturadas ao augmentation das outras, mas elas mesmas nunca são aumentadas (tratadas como skip_labels). Continuam sendo classes treináveis. |
| `keep_original` | `true` | Mantém a amostra limpa junto das cópias aumentadas. |
| `augment_test` | `false` | Aplica o aumento também ao conjunto de teste, além do conjunto de treino. |
| `skip_labels` | `null` | Rótulos que não recebem aumento. Continuam sendo classes treináveis. |
| `concatenate_augmentation_dir` | `false` | Quando true, todos os arquivos de augmentation_dir são concatenados em um único array em memória, usado como única fonte de ruído. Se houver um arquivo de rótulos .txt do Audacity ao lado de um áudio, apenas os trechos rotulados são usados; caso contrário, o arquivo inteiro é incluído. |
| `random_augmentation_dir` | `false` | Quando true, os arquivos de augmentation_dir são embaralhados uma vez (usando random_seed) e atribuídos em rodízio às amostras aumentadas: um arquivo por par (amostra, SNR), sem repetição dentro de cada passagem. Assim como concatenate_augmentation_dir, produz uma única condição de ruído, mas a partir de arquivos individuais em vez de uma faixa única. |

## `training` — Treinamento

Seleção do método de treinamento e ajustes de parâmetros específicos.

| Key | Padrão | Descrição |
|---|---|---|
| `classifier` | `keras` | Qual cabeça de classificação treinar: keras, sklearn ou ambas. |

### `training.keras` — Keras

*Somente quando `training.classifier` = `keras` / `both`.*

| Key | Padrão | Descrição |
|---|---|---|
| `hidden_units` | `256` | Unidades da camada Dense oculta. 0 significa sem camada oculta (classificador linear). |
| `dropout` | `0.25` | Taxa de dropout antes de cada camada Dense. |
| `epochs` | `50` | Número máximo de épocas. O early stopping pode encerrar o treino antes. |
| `batch_size` | `32` | Tamanho do mini-batch. |
| `learning_rate` | `0.0001` | Taxa de aprendizado de pico do Adam (decaimento cosseno com aquecimento linear). |
| `output_activation` | `null` | Ativação de saída: null (treinado com softmax, exporta logits; o padrão), "sigmoid", "softmax" ou "grouped_softmax" (requer label_groups). |
| `label_groups` | `{}` | Softmax agrupado: um softmax dentro de cada grupo, com os grupos independentes entre si. Associa cada nome de grupo aos seus rótulos membros, e cada grupo ganha uma coluna de saída sintética "<grupo>_none". Membros de um mesmo grupo são mutuamente exclusivos; membros de grupos diferentes podem ser previstos juntos. Rótulos de treino que não estão em nenhum grupo são fundo: não ganham coluna de saída e fornecem o alvo "none" de todos os grupos. |
| `export_logits` | `false` | Remove a camada de ativação de saída antes da exportação, para que o classificador exportado emita logits brutos enquanto o treino usou output_activation (como faz o BirdNET-Analyzer com classifier.pop()). Não tem efeito quando output_activation é null. Quem usar o modelo deve aplicar a ativação. É recusado junto com exclude_labels em um classificador softmax ou agrupado, em que os logits excluídos são necessários para reconstruir o softmax (output_activation null com exclude_labels é recusado pelo mesmo motivo). |
| `normalize_embeddings` | `true` | Normalização z-score dos embeddings de entrada, ajustada no conjunto de treino. |
| `focal_loss` | `false` | Usa focal loss em vez de entropia cruzada. |
| `focal_loss_gamma` | `2.0` | Parâmetro de foco γ da focal loss. |
| `focal_loss_alpha` | `0.25` | Parâmetro de balanceamento de classes α da focal loss. |
| `label_smoothing` | `false` | Aplica suavização de rótulos aos alvos one-hot antes do treino. |
| `label_smoothing_alpha` | `0.1` | Valor subtraído dos rótulos positivos e redistribuído aos negativos. |
| `mixup` | `false` | Aplica aumento de dados mixup aos embeddings de treino. |
| `mixup_ratio` | `0.25` | Fração das amostras positivas a misturar. |
| `mixup_alpha` | `0.2` | Parâmetro da distribuição Beta do coeficiente de mistura. |
| `upsampling_ratio` | `0.0` | Proporção de upsampling das classes minoritárias antes do treino. |
| `upsampling_mode` | `repeat` | Método de upsampling: repeat (duplicação aleatória), mean (média entre pares), linear (interpolação aleatória) ou smote (interpolação por k-NN). |
| `seed` | `true` | Fixa as sementes aleatórias do TensorFlow e do NumPy a partir de dataset.random_seed antes do treino, para reprodutibilidade. Use false para desativar; os resultados do treino passam então a variar entre execuções. |

### `training.sklearn` — scikit-learn

*Somente quando `training.classifier` = `sklearn` / `both`.*

| Key | Padrão | Descrição |
|---|---|---|
| `C` | `1.0` | Força inversa da regularização. |
| `max_iter` | `2000` | Número máximo de iterações do solver. |
| `solver` | `lbfgs` | Algoritmo solver da LogisticRegression. |

### `training.audio_mixup` — Mixup de áudio

| Key | Padrão | Descrição |
|---|---|---|
| `enabled` | `true` | Mistura o áudio de janelas de treino de classes diferentes, gera o embedding da mistura e treina com a união dos rótulos. Requer uma cabeça Keras sigmoid. |
| `labels` | `[]` | Rótulos cujas janelas podem ser misturadas. Janelas de qualquer outro rótulo nunca são usadas como fonte, embora continuem sendo treinadas como amostras reais. Vazio significa todos os rótulos. |
| `n_mixes` | `null` | Número de misturas. Substitui ratio quando definido. |
| `ratio` | `0.5` | Número de misturas como fração das janelas reais de treino. |
| `max_sources` | `2` | Número máximo de fontes por mistura: 2, ou 3 (ver p_three_sources). |
| `p_three_sources` | `0.0` | Proporção de misturas com três fontes quando max_sources é 3. |
| `snr_db` | `[-6.0, 6.0]` | Faixa de nível [mín, máx] em dB de cada fonte adicionada, em relação à primeira. |
| `pairing` | `balanced` | balanced: sorteia uma classe uniformemente e depois um clipe dela; uniform: sorteia clipes uniformemente. |
| `exclusive_groups` | `{}` | Grupos de rótulos mutuamente exclusivos (por exemplo, tipos de canto ou níveis de intensidade de uma espécie) que nunca são misturados entre si. |
| `background_labels` | `[]` | Rótulos que representam o fundo, e não espécies. Uma mistura contém no máximo um deles, e sempre ao menos uma fonte que não é de fundo. |
| `background_target` | `drop` | drop: uma fonte de fundo não acrescenta rótulo ao alvo da mistura; include: acrescenta o próprio rótulo. |
| `seed` | `null` | Semente para sortear as misturas. O padrão é dataset.random_seed. |
| `test_mix_ratio` | `0.0` | Também gera misturas apenas a partir de janelas de teste, como fração das janelas de teste que podem ser misturadas (0 significa nenhuma). São relatadas em uma seção separada e nunca somadas ao conjunto de teste real. |

## `output` — Parâmetros de saída

Configurações de caminho, formato e precisão para os arquivos de saída.

| Key | Padrão | Descrição |
|---|---|---|
| `output_path` | `./outputs` | Diretório principal de toda a saída. |
| `model_name` | `custom_classifier` | Nome do modelo, usado nos nomes dos arquivos e no subdiretório de saída. |
| `model_version` | `1.0` | String de versão usada nos nomes dos arquivos. |
| `output_type` | `head` | O que exportar: head (apenas o classificador), full (modelo base e classificador combinados) ou both (ambos). |
| `output_format` | `onnx` | Formato de arquivo do modelo exportado. |
| `exclude_labels` | `[]` | Rótulos a excluir da saída do classificador exportado. Continuam sendo usados no treino. Exige um classificador Keras que exporte probabilidades ou logits por classe, por isso é recusado com output_activation null, ou com softmax e export_logits. |
| `export_dataset` | `false` | Exporta os segmentos de áudio como arquivos WAV em subpastas por rótulo. |
| `export_mixes` | `false` | Exporta apenas as misturas de áudio (training.audio_mixup) como arquivos WAV em dataset/mixes/, para conferi-las quando as janelas reais já foram exportadas. |
| `export_embeddings` | `false` | Salva os embeddings calculados para reuso. |
| `embeddings_format` | `npy` | Formato de armazenamento dos embeddings exportados e em cache: npy (um arquivo por amostra) ou sqlite (um único arquivo .db por execução, mais portátil). |
| `embeddings_path` | `null` | Diretório (npy) ou caminho de arquivo (sqlite) dos embeddings exportados. O padrão é <output_dir>/embeddings ou <output_dir>/embeddings.db. |
| `embeddings_overwrite` | `false` | Aplica-se apenas ao modo --embeddings. Quando já existe um repositório de embeddings no caminho resolvido, recalcula e sobrescreve. Quando false (o padrão), o repositório existente é reaproveitado: embeddings já armazenados nunca são recalculados, e apenas as amostras que faltam são calculadas e adicionadas. |
| `head_path` | `null` | Caminho de um classificador ONNX já existente para --merge (sem treino). |
| `extract_from` | `null` | Caminho de um modelo completo do BirdNET-Analyzer (.tflite) para --extract_head. O classificador final é lido do flatbuffer e reexportado como modelo apenas de classificador, sem conversão do backbone nem treino. |
| `labels_file` | `null` | Arquivo opcional com os rótulos das classes para --extract_head, com um rótulo por linha. Se omitido, é usado um arquivo ``<model>_Labels.txt`` ao lado do modelo, quando existir. |
| `data_types` | `null` | Precisões a exportar: um subconjunto de {"FP32", "FP16", "INT8"}. Quando não definido, usa [foundation_model.data_type]. Cada precisão gera um arquivo exportado separado, com a precisão no nome. INT8 usa quantização dinâmica (apenas dos pesos), sem dataset de calibração. |

## `umap` — UMAP

Ajustes da projeção dos embeddings. Requer o extra [umap].

| Key | Padrão | Descrição |
|---|---|---|
| `enabled` | `false` | Calcula a projeção e grava o CSV e os gráficos. Requer o extra [umap]. |
| `n_neighbors` | `15` | Tamanho da vizinhança: valores baixos preservam a estrutura local; valores altos, a global. |
| `min_dist` | `0.1` | Quão próximos os pontos podem ficar. |
| `n_components` | `2` | Número de dimensões da projeção. |
| `metric` | `euclidean` | Métrica de distância entre embeddings. |
| `random_seed` | `null` | Semente aleatória do UMAP. Usa dataset.random_seed quando não definida. |
| `cache_csv` | `null` | Caminho de um CSV de dados do UMAP já gravado. Quando definido e o arquivo existe, a execução de embeddings dispensa todo o cálculo (carga do dataset, extração de embeddings, KMeans e ajuste do UMAP) e apenas redesenha os gráficos a partir das coordenadas em cache. Isso é útil para ajustar o estilo dos gráficos sem recalcular. |

<!-- /generated:config-reference -->
