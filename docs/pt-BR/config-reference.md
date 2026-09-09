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
| `registry_id` | `null` | Carrega todos os padrões de um backbone conhecido em uma linha. Qualquer campo definido junto sobrescreve o padrão do registro. |
| `name` | *obrigatório* | Usado nos relatórios e nos nomes dos arquivos de saída |
| `version` | `unknown` | String de versão do modelo |
| `data_type` | `FP32` | Precisão dos pesos, por exemplo FP32 / INT8 |
| `format` | `onnx` | Formato do modelo em disco |
| `source` | `local` | Origem: caminho local, hub huggingface ou kaggle |
| `path` | `null` | caminho do arquivo/diretório local do modelo |
| `hf_repo` | `null` | ex.: "biodiversica/birdnet" |
| `hf_filename` | `null` | arquivo específico dentro do repositório HF |
| `hf_revision` | `null` | branch / tag / commit |
| `kaggle_handle` | `null` | ex.: "google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier" |
| `kaggle_filename` | `null` | arquivo específico dentro do diretório baixado (onnx/tflite) |
| `sample_rate` | `48000` | Taxa de amostragem esperada pelo backbone, em Hz |
| `window_samples` | `null` | tem prioridade sobre window_seconds |
| `window_seconds` | `null` | ex.: 3.0 |
| `input_name` | `input` | Nome do tensor de entrada de áudio |
| `output_name` | `embedding` | Nome do tensor de saída (o embedding) |
| `embedding_size` | `1024` | Dimensão do vetor de embedding |
| `onnx_providers` | `null` | Execution providers do ONNX Runtime (ex.: ["CUDAExecutionProvider", "CPUExecutionProvider"]). Com None, segue a ordem de prioridade do próprio ORT. |
| `onnx_batch_size` | `1` | Número de janelas de áudio processadas em uma única chamada de inferência ONNX. Valores > 1 ativam o modo de lote na GPU (usado só com format="onnx"). |
| `tflite_output_tensor_offset` | `0` | Deslocamento aplicado ao índice do tensor de saída TFLite para chegar ao tensor de embedding. Use -1 para modelos como o BirdNET tflite, em que o classificador é a primeira saída e o embedding fica uma posição antes dele na lista de tensores do modelo. |
| `tflite_trim_to_embedding` | `true` | Corta o grafo TFLite até o tensor de embedding quando o modelo calcula mais do que isso (um classificador embutido, saídas auxiliares). Feito uma vez ao carregar; todas as janelas seguintes já pulam os ramos descartados. Use false para rodar o modelo exatamente como ele foi distribuído. |

## `dataset` — Dataset

Seleção de fontes de áudio, tipos de rótulos, pré-processamento e criação do dataset para treinamento do modelo.

| Key | Padrão | Descrição |
|---|---|---|
| `data_dir` | `[]` | Fonte(s) de áudio local; pode ficar vazio quando só ext_table_file é usado |
| `label_mode` | `subfolders` | Como os rótulos estão organizados em data_dir |
| `table_file` | `null` | CSV/TSV; usado quando label_mode="table" |
| `audio_extensions` | `["wav", "flac", "mp3", "ogg"]` | Extensões de áudio aceitas (sem diferenciar maiúsculas) — **nível da execução** (não pode ser sobrescrito por fonte) |
| `overlap` | `0.0` | Sobreposição entre janelas consecutivas ao criar segmentos rotulados (0.0–1.0) |
| `embedding_workers` | `4` | Workers paralelos da extração de embeddings (limitado aos núcleos de CPU disponíveis) — **nível da execução** (não pode ser sobrescrito por fonte) |
| `embeddings_cache_path` | `null` | Caminho onde procurar embeddings .npy pré-computados antes de rodar o modelo — **nível da execução** (não pode ser sobrescrito por fonte) |
| `append_dataset_path` | `null` | Caminho de um dataset já exportado (subpastas com train/test). Amostras novas de data_dir que ainda não estejam nele são acrescentadas. — **nível da execução** (não pode ser sobrescrito por fonte) |
| `test_ratio` | `0.2` | Divisão treino / teste; ignorada quando o dataset já traz a divisão — **nível da execução** (não pode ser sobrescrito por fonte) |
| `random_seed` | `42` | Semente para divisões, embaralhamento e deslocamentos de ruído — **nível da execução** (não pode ser sobrescrito por fonte) |
| `filename_col` | `filename` | Nomes de coluna usados nos modos table / ext_table_file |
| `label_col` | `label` | Rótulo da classe na linha |
| `start_col` | `start_time` | Início do trecho, em segundos |
| `end_col` | `end_time` | Fim do trecho, em segundos |
| `split_col` | `split` | opcional; valores "train" / "test" |
| `ext_table_file` | `null` | Tabela do iNaturalist — CSV/TSV com linhas de observation_id (ou misturadas com linhas de filename) |
| `obs_id_col` | `observation_id` | Id da observação no iNaturalist |
| `sound_index_col` | `sound_index` | Índice do som no iNaturalist (a partir de 0) |
| `xc_id_col` | `xc_id` | Id da gravação no Xeno-canto |
| `ext_cache_dir` | `null` | Cache local do áudio remoto baixado; padrão ~/.cache/bioaccx/ext — **nível da execução** (não pode ser sobrescrito por fonte) |
| `xc_api_key` | `null` | Chave da API v3 do Xeno-canto — necessária para os metadados (busca do nome científico). O áudio pode ser baixado sem chave pela URL direta. Cadastre-se em https://xeno-canto.org/explore/api — **secreto** (nunca devolvido ao navegador) — **nível da execução** (não pode ser sobrescrito por fonte) |
| `arbimon_credentials_path` | `null` | Arbimon / rfcx — caminho do arquivo de credenciais gerado por rfcx.Client().authenticate(persisted_credentials_path=...). Os nomes de coluna a seguir identificam as linhas do Arbimon em ext_table_file. — **nível da execução** (não pode ser sobrescrito por fonte) |
| `arbimon_stream_id_col` | `stream_id` | Id do stream / site no Arbimon |
| `arbimon_date_col` | `date` | Data local da gravação |
| `arbimon_time_col` | `time` | Hora local de início da gravação |
| `arbimon_utc_offset_col` | `utc_offset` | Fuso horário (offset UTC) dessas datas e horas |
| `filter` | `null` | Pré-processamento do áudio aplicado antes da divisão em segmentos (filtro → velocidade → segmentos). filter: 'hpf' \| 'lpf' \| 'bpf' \| null; filter_freq: valor em Hz para hpf/lpf, lista [hz_baixo, hz_alto] para bpf |
| `filter_freq` | `null` | Frequência de corte em Hz para hpf/lpf; [low_hz, high_hz] para bpf |
| `filter_order` | `5` | Ordem do filtro Butterworth |
| `speed` | `1.0` | Multiplicador da velocidade de reprodução (>1 mais rápido / mais curto, <1 mais lento / mais longo). Os tempos dos rótulos são escalados junto: novo_tempo = tempo_antigo / speed. |
| `ssh_host` | `null` | Acesso SSH / SFTP — com ssh_host definido, os caminhos de data_dir são tratados como caminhos remotos no servidor SSH e espelhados localmente via paramiko antes de a pipeline rodar. Requer: pip install paramiko |
| `ssh_user` | `null` | Usuário SSH; por padrão, o usuário local |
| `ssh_port` | `22` | Porta SSH |
| `ssh_key_path` | `null` | caminho do arquivo de chave privada |
| `random_sample_shift` | `false` | Com True, amostras mais curtas que a janela do modelo base são posicionadas em um deslocamento aleatório dentro da janela, em vez de começarem sempre na posição 0. Cada cópia aumentada da mesma amostra recebe um deslocamento diferente. |
| `min_anchor_fraction` | `0.1` | Fração mínima da janela que precisa ser áudio novo (ainda não coberto) para que o segmento âncora no fim do arquivo seja emitido. Evita segmentos quase duplicados quando os arquivos são só um pouco maiores que a janela (ex.: 3,013 s com janela de 3 s). Use 0.0 para sempre emitir o segmento âncora. |
| `sources` | `null` | Combina vários caminhos com configurações específicas para juntar todos em uma execução única para a criação do dataset. Cada bloco de fonte aqui adicionado herda os campos do bloco principal e os sobrescreve com os seus. Campos de 'NÍVEL DA EXECUÇÃO' nunca são sobrescritos por parâmetros de uma fonte adicional. |

### `dataset.augmentation` — Augmentation

| Key | Padrão | Descrição |
|---|---|---|
| `snr_levels` | *obrigatório* | Valores de SNR em dB; uma cópia aumentada por arquivo de ruído por nível |
| `augmentation_dir` | `null` | Diretório de arquivos WAV usados como fontes de ruído. Opcional quando augmentation_labels está definido; se os dois forem dados, o conjunto de ruído é a união. |
| `augmentation_labels` | `null` | Rótulos do dataset em criação cujo áudio é usado como fonte adicional de ruído (além de augmentation_dir). As amostras desses rótulos são misturadas ao augmentation das outras, mas elas mesmas nunca são aumentadas (tratadas como skip_labels). Continuam sendo classes treináveis. |
| `keep_original` | `true` | Mantém a amostra limpa junto das cópias aumentadas |
| `augment_test` | `false` | Aplica o aumento também ao conjunto de teste, não só ao de treino |
| `skip_labels` | `null` | Rótulos que não recebem aumento; seguem sendo classes treináveis |
| `concatenate_augmentation_dir` | `false` | Com True, todos os arquivos de augmentation_dir são concatenados em um único array em memória usado como única fonte de ruído. Se existir um arquivo de rótulos .txt do Audacity ao lado de um áudio, só os trechos rotulados são usados; caso contrário, o arquivo inteiro entra. |
| `random_augmentation_dir` | `false` | Com True, os arquivos de augmentation_dir são embaralhados uma vez (usando random_seed) e atribuídos em rodízio às amostras aumentadas — um arquivo por par (amostra, SNR), sem repetição dentro de cada passagem. Produz uma condição de ruído (como concatenate_augmentation_dir), mas a partir de arquivos individuais em vez de uma faixa única. |

## `training` — Treinamento

Seleção do método de treinamento e ajustes de parâmetros específicos.

| Key | Padrão | Descrição |
|---|---|---|
| `classifier` | `keras` | Qual head treinar: keras, sklearn ou ambos |

### `training.keras` — Keras

*Somente quando `training.classifier` = `keras` / `both`.*

| Key | Padrão | Descrição |
|---|---|---|
| `hidden_units` | `256` | Unidades da camada Dense oculta; 0 = sem camada oculta (classificador linear) |
| `dropout` | `0.25` | Taxa de dropout antes de cada camada Dense |
| `epochs` | `50` | Máximo de épocas; o early stopping pode parar antes |
| `batch_size` | `32` | Tamanho do mini-batch |
| `learning_rate` | `0.0001` | Taxa de aprendizado de pico do Adam (decaimento cosseno com warmup linear) |
| `output_activation` | `null` | Ativação de saída: None (logits, padrão) \| "sigmoid" \| "softmax" \| "grouped_softmax" (requer label_groups) |
| `label_groups` | `{}` | Softmax agrupado: softmax dentro de cada grupo, grupos independentes entre si. Mapeia nome do grupo -> rótulos membros; cada grupo ganha uma coluna de saída sintética "<grupo>_none". Membros de um mesmo grupo são mutuamente exclusivos; membros de grupos diferentes podem disparar juntos. Rótulos de treino que não estão em nenhum grupo são fundo: não ganham coluna de saída e fornecem o alvo "none" de todos os grupos. |
| `export_logits` | `false` | Remove a camada de ativação de saída antes de exportar, para que o classificador exportado emita logits enquanto o treino usou output_activation (é o que o BirdNET-Analyzer faz com classifier.pop()). Não faz nada quando output_activation é None. Quem consumir a saída precisa aplicar a ativação. |
| `normalize_embeddings` | `true` | Normalização z-score dos embeddings de entrada (ajustada em X_train) |
| `focal_loss` | `false` | Focal loss (substitui a entropia cruzada quando ativada) |
| `focal_loss_gamma` | `2.0` | Parâmetro de foco γ da focal loss |
| `focal_loss_alpha` | `0.25` | Parâmetro de balanceamento de classes α da focal loss |
| `label_smoothing` | `false` | Suavização de rótulos aplicada aos alvos one-hot antes do treino |
| `label_smoothing_alpha` | `0.1` | Valor subtraído dos rótulos positivos e redistribuído aos negativos |
| `mixup` | `false` | Aumento de dados mixup nos embeddings de treino |
| `mixup_ratio` | `0.25` | Fração das amostras positivas a misturar |
| `mixup_alpha` | `0.2` | Parâmetro Beta do coeficiente de mistura |
| `upsampling_ratio` | `0.0` | Upsampling das classes minoritárias antes do treino |
| `upsampling_mode` | `repeat` | repeat = duplicação aleatória, mean = média entre pares, linear = interpolação aleatória, smote = interpolação por k-NN |
| `seed` | `true` | Fixa as sementes aleatórias do TF e do numpy antes do treino, para reprodutibilidade. Usa dataset.random_seed. Use false para desativar (o treino varia de execução para execução). |

### `training.sklearn` — sklearn

*Somente quando `training.classifier` = `sklearn` / `both`.*

| Key | Padrão | Descrição |
|---|---|---|
| `C` | `1.0` | Força inversa da regularização |
| `max_iter` | `2000` | Máximo de iterações do solver |
| `solver` | `lbfgs` | Algoritmo solver da LogisticRegression |

## `output` — Parâmetros de saída

Configurações de caminho, formato e precisão para os arquivos de saída.

| Key | Padrão | Descrição |
|---|---|---|
| `output_path` | `./outputs` | Diretório onde toda a saída é escrita |
| `model_name` | `custom_classifier` | Usado nos nomes dos arquivos e no subdiretório de saída |
| `model_version` | `1.0` | String de versão usada nos nomes dos arquivos |
| `output_type` | `head` | head = só o classificador; full = modelo base + classificador; both = salva os dois |
| `output_format` | `onnx` | Formato(s) de arquivo do modelo exportado |
| `exclude_labels` | `[]` | Rótulos a excluir da saída do classificador exportado (ainda usados no treino) |
| `export_dataset` | `false` | Exporta os segmentos de áudio como arquivos WAV em subpastas por rótulo |
| `export_embeddings` | `false` | Salva os embeddings calculados para reuso |
| `embeddings_format` | `npy` | Formato de armazenamento dos embeddings exportados/em cache: npy (um arquivo por amostra) ou sqlite (um único .db por execução, mais portátil) |
| `embeddings_path` | `null` | Diretório (npy) ou caminho de arquivo (sqlite) dos embeddings exportados; padrão <output_dir>/embeddings ou <output_dir>/embeddings.db |
| `embeddings_overwrite` | `false` | Só no modo --embeddings: quando já existe um repositório de embeddings no caminho resolvido, recalcula e sobrescreve. Com False (padrão), o repositório existente é reaproveitado como está — nenhum embedding é recalculado e a execução apenas (re)constrói as saídas do UMAP (quando umap.enabled). |
| `head_path` | `null` | Caminho de um classificador ONNX já existente para --merge (sem treino) |
| `extract_from` | `null` | Caminho de um modelo completo do BirdNET-Analyzer (.tflite) para --extract_head: classificador final é lido do flatbuffer e reexportado como modelo só-classificador (sem conversão do backbone, sem treino). |
| `labels_file` | `null` | Arquivo opcional com os rótulos das classes para --extract_head (um rótulo por linha). Se omitido, usa um ``<model>_Labels.txt`` ao lado, quando existir. |
| `data_types` | `null` | Precisões a exportar: subconjunto de {"FP32", "FP16", "INT8"}. None → usa [foundation_model.data_type]. Cada precisão gera um arquivo exportado separado, com a precisão no nome. INT8 usa quantização dinâmica / só de pesos (sem dataset de calibração). |

## `umap` — UMAP

Ajustes da projeção dos embeddings. Requer o extra [umap].

| Key | Padrão | Descrição |
|---|---|---|
| `enabled` | `false` | Calcula a projeção e escreve o CSV e os gráficos (requer o extra [umap]) |
| `n_neighbors` | `15` | Tamanho da vizinhança: valores baixos preservam a estrutura local, altos a global |
| `min_dist` | `0.1` | Quão próximos os pontos podem ficar |
| `n_components` | `2` | Número de dimensões da projeção |
| `metric` | `euclidean` | Métrica de distância entre embeddings |
| `random_seed` | `null` | Semente aleatória do UMAP; usa dataset.random_seed quando None. |
| `cache_csv` | `null` | Caminho de um CSV de dados do UMAP já gravado. Quando definido e o arquivo existe, a execução de embeddings pula todo o cálculo (carga do dataset, extração de embeddings, KMeans, ajuste do UMAP) e só redesenha os gráficos a partir das coordenadas em cache — útil para ajustar o estilo dos gráficos sem recalcular. |

<!-- /generated:config-reference -->
