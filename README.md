# Wound Segmentation — Pipeline de Treinamento e Comparação de CNNs

Solução completa para treinamento, validação, avaliação e comparação de quatro
abordagens de segmentação semântica aplicadas a imagens de feridas.

---

## 1. Onde colocar as imagens (estrutura da pasta `data/`)

Esta é a parte mais importante para você começar a usar o projeto. A pasta
`data/` deve separar o conjunto usado no treinamento do conjunto reservado
para avaliação final:

```text
data/
│
├── treino/
│   ├── imagens/               <- imagens usadas no treinamento
│   └── mascaras/              <- máscaras correspondentes
│
├── validacao/
│   ├── imagens/               <- imagens usadas somente na avaliação
│   └── mascaras/              <- máscaras correspondentes
│
└── processed/                 <- cache opcional
```

### Regra de pareamento

Uma imagem em `data/treino/imagens/ferida_001.jpg` ou
`data/validacao/imagens/ferida_001.jpeg` precisa ter uma máscara com o mesmo
nome-base em `mascaras/`, por exemplo `ferida_001.png`. O sistema
(`datasets/wound_dataset.py`)
varre `imagens/`, procura o arquivo de mesmo nome em `mascaras/` e **ignora
automaticamente** qualquer imagem que não tenha máscara correspondente.

---

## 2. Visão Geral das Quatro Pipelines

| # | Nome interno | Pré-processamento OpenCV | Encoder | Transfer Learning |
|---|----------|--------------------|---------|--------------------|
| 1 | `unet_resnet50` | Não | ResNet-50 (ImageNet) | Sim |
| 2 | `unet` | Não | Construído do zero | Não |
| 3 | `unet_resnet50_opencv` | Sim | ResNet-50 (ImageNet) | Sim |
| 4 | `unet_opencv` | Sim | Construído do zero | Não |

O treinamento lê exclusivamente `data/treino/`. Os scripts de avaliação leem
exclusivamente `data/validacao/`.

---

## 3. Estrutura Completa do Repositório

```text
wound_segmentation/
│
├── data/                        # Seus dados (ver seção 1)
│   ├── treino/
│   │   ├── imagens/
│   │   └── mascaras/
│   ├── validacao/
│   │   ├── imagens/
│   │   └── mascaras/
│   └── processed/
│
├── configs/
│   └── train_config.yaml        # Configuração central: caminhos, hiperparâmetros,
│                                 # toggles de pré-processamento e augmentation
│
├── datasets/
│   ├── wound_dataset.py         # Classe WoundDataset + listagem de pares imagem/máscara
│   ├── augmentations.py         # Monta o pipeline Albumentations a partir do YAML
│   └── data_split.py            # Divisão treino/validação/teste (reprodutível)
│
├── preprocessing/
│   └── opencv_pipeline.py       # As 10 técnicas OpenCV, cada uma ativável/desativável
│
├── models/
│   ├── unet.py                  # U-Net clássica, implementação própria (sem backbone)
│   └── unet_resnet50.py         # U-Net com encoder ResNet-50 pré-treinado
│
├── training/
│   ├── trainer.py               # Loop de treino/validação (mixed precision, early
│   │                             # stopping, checkpoints, scheduler, TensorBoard)
│   ├── losses.py                # BCE, Dice, BCE+Dice, Focal, Tversky
│   └── metrics.py               # Dice, IoU, Precisão, Recall, F1, Acurácia
│
├── evaluation/
│   └── evaluator.py             # Avaliação no conjunto de validação + relatório
│
├── inference/
│   └── predict.py               # Inferência via linha de comando em uma nova imagem
│
├── utils/
│   ├── logger.py                # Logging estruturado (console + arquivo)
│   ├── seed.py                  # Reprodutibilidade + detecção automática de dispositivo
│   └── visualization.py         # Gráficos: curvas, matriz de confusão, comparações visuais
│
├── experiments/                 # Configuração + hiperparâmetros salvos por pipeline
│   ├── unet/
│   ├── unet_resnet50/
│   ├── unet_opencv/
│   └── unet_resnet50_opencv/
│
├── checkpoints/                 # Pesos salvos (*_best.pth e *_last.pth) por pipeline
├── logs/                        # Logs em texto + logs do TensorBoard
├── results/                     # Métricas (CSV/JSON), gráficos e relatório comparativo
│
├── train.py                     # Script principal de treinamento
├── evaluate.py                  # Script principal de avaliação e comparação final
└── requirements.txt
```

### Como cada pasta se relaciona no fluxo de execução

1. Você coloca os dados de treinamento em `data/treino/imagens` e
   `data/treino/mascaras`, e os dados de avaliação em `data/validacao/imagens`
   e `data/validacao/mascaras`.
2. `train.py` lê `configs/train_config.yaml`, monta os datasets
   (`datasets/`), aplica pré-processamento OpenCV se a pipeline exigir
   (`preprocessing/`), aplica augmentation e treina o modelo (`models/` +
   `training/`).
3. Durante o treino, `training/trainer.py` grava logs em `logs/`, salva
   checkpoints em `checkpoints/` e a configuração usada em `experiments/<pipeline>/`.
4. `evaluate.py` carrega os checkpoints `_best.pth`, avalia o conjunto
   `data/validacao/` (`evaluation/`) e grava métricas e gráficos em `results/`.
5. `inference/predict.py` usa um checkpoint já treinado para gerar a máscara
   de uma imagem nova, fora do fluxo de treino.

---

## 4. Fluxograma da Solução

## 5. Avaliação em lote e medição da ferida

Os módulos [`evaluate_segmentation.py`](./evaluate_segmentation.py) e
[`wound_measurement_pipeline.py`](./wound_measurement_pipeline.py) são
separados do treinamento e aceitam os checkpoints já produzidos por
`training/trainer.py`. Eles reconhecem `png`, `jpg`, `jpeg`, `bmp`, `tif` e
`tiff`; a correspondência entre imagem e máscara manual é feita pelo nome-base
do arquivo, mesmo quando as extensões são diferentes.

### Inferência e comparação

```powershell
python evaluate_segmentation.py `
  --images_dir data/validacao/imagens `
  --masks_dir data/validacao/mascaras `
  --weights checkpoints/unet_best.pth `
  --architecture unet `
  --output_dir results/evaluation `
  --batch_size 4
```

Para uma imagem use `--image` e `--mask` (a máscara deve possuir o mesmo
nome-base). A máscara salva em `predicted_masks/` é binária: fundo `0` e
ferida `255`. O CSV contém Dice, IoU/Jaccard, acurácia, precisão, recall,
especificidade, F1, matriz de contagens e áreas, seguido por uma linha de
resumo com média, mediana e desvio-padrão. Em avaliação em lote, imagens sem
máscara correspondente são ignoradas e registradas como aviso em
`evaluation.log`, sem interromper a execução.

### Busca do limiar

```powershell
python evaluate_segmentation.py `
  --images_dir data/validacao/imagens `
  --masks_dir data/validacao/mascaras `
  --weights checkpoints/unet_best.pth `
  --architecture unet `
  --output_dir results/threshold `
  --find_best_threshold --threshold_start 0.10 --threshold_end 0.90 `
  --threshold_step 0.05 --threshold_metric dice
```

São aceitos `dice`, `iou`, `acuracia` e `f1`. O resultado é gravado em
`best_threshold.json` e `threshold_search/threshold_results.csv`, e o limiar
selecionado é usado nas máscaras finais. A saída da rede é tratada como
logit quando contém valores fora de `[0, 1]`; caso contrário é tratada como
probabilidade.

### Pipeline completa com RulerNet

Baixe o `model.onnx` oficial de
[`ymp5078/RulerNet`](https://huggingface.co/ymp5078/RulerNet) (a alternativa
CPU documentada no model card) e execute:

```powershell
python wound_measurement_pipeline.py `
  --images_dir data/validacao/imagens `
  --manual_masks_dir data/validacao/mascaras `
  --segmentation_weights checkpoints/unet_best.pth `
  --architecture unet `
  --rulernet_model caminho/para/model.onnx `
  --output_dir results/measurements `
  --threshold 0.45
```

O adaptador usa a entrada RGB `(1, 3, 768, 768)` e reconstrói os pontos a
partir das cinco saídas do ONNX, como no demo oficial. A escala é a mediana
das distâncias entre marcas, em pixels/cm. Assim, `area_cm2 =
area_pixels / pixels_por_cm²`; sem régua detectável, os campos em centímetros
permanecem nulos. Os arquivos principais são `predicted_masks/`,
`measurement_results.csv` e `measurements/results.json`.

As medidas em centímetros pressupõem que a régua e a ferida estão no mesmo
plano. Perspectiva, inclinação, oclusão e uma régua fora do plano podem
introduzir erro. `onnxruntime` e `huggingface_hub` são dependências opcionais
para esta etapa; o treinamento não depende delas.

### Testes

```powershell
pytest tests/test_evaluation.py
```

```text
                 ┌─────────────────────────────────────────┐
                 │      data/treino/imagens + mascaras       │
                 │        (únicos dados usados no treino)     │
                 └──────────────────┬─────────────────────┘
                                    │
                 ┌──────────────────▼─────────────────────┐
                 │        Divisão Estratificada             │
                 │  Treino 70% / Validação 15% / Teste 15% │
                 └──────────────────┬─────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
 ┌──────────▼─────────┐  ┌──────────▼─────────┐  ┌──────────▼─────────┐
 │  Pipelines 1 e 2     │  │  Pipelines 3 e 4     │  │   (mesma divisão,   │
 │  (sem OpenCV)        │  │  (com OpenCV)         │  │   usada nas 4)      │
 └──────────┬───────────┘  └──────────┬───────────┘  └─────────────────────┘
            │                          │
 ┌──────────▼───────────┐   ┌──────────▼───────────┐
 │ Data Augmentation      │   │ OpenCV Pipeline         │
 │  (Albumentations)      │   │ (CLAHE, ruído, etc.)   │
 └──────────┬───────────┘   └──────────┬───────────┘
            │                          │
            │               ┌──────────▼───────────┐
            │               │ Data Augmentation      │
            │               │  (Albumentations)      │
            │               └──────────┬───────────┘
            │                          │
 ┌──────────▼──────────────────────────▼───────────┐
 │            Trainer (training/trainer.py)          │
 │  Mixed Precision • Grad Clipping • LR Scheduler   │
 │  Early Stopping • Checkpoint • TensorBoard         │
 └──────────────────────┬─────────────────────────────┘
                         │
            ┌────────────▼────────────┐
            │  checkpoints/*_best.pth  │
            │  checkpoints/*_last.pth  │
            └────────────┬────────────┘
                         │
            ┌────────────▼────────────┐
            │ Evaluator (validation set)│
            │  Dice, IoU, Precisão,    │
            │  Recall, F1, Acurácia    │
            └────────────┬────────────┘
                         │
            ┌────────────▼────────────┐
            │  Relatório Comparativo   │
            │  results/*.csv / *.json  │
            │  results/*.png           │
            └───────────────────────────┘
```

---

## 5. Instruções de Execução

### 5.1. Instalação

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 5.2. Organizar os dados

Siga a seção 1 deste README. O treinamento utiliza somente
`data/treino/imagens/` e `data/treino/mascaras/`. A avaliação final utiliza
somente `data/validacao/imagens/` e `data/validacao/mascaras/`.

### 5.3. Ajustar a configuração

Edite `configs/train_config.yaml` para ativar/desativar pipelines, técnicas
de pré-processamento OpenCV, augmentations, hiperparâmetros de treino e a
função de perda. O treinamento usa `caminhos.imagens_treino` e
`caminhos.mascaras_treino`; a avaliação usa `caminhos.imagens_validacao` e
`caminhos.mascaras_validacao`.

### 5.4. Treinar

```bash
# Treinar todas as quatro pipelines sequencialmente
python3 train.py --config configs/train_config.yaml

# Treinar apenas uma pipeline específica
python3 train.py --config configs/train_config.yaml --pipeline unet_resnet50_opencv

# Retomar uma pipeline a partir do último checkpoint salvo
python train.py --config configs/train_config.yaml --pipeline unet_resnet50 --retomar

# Retomar usando um checkpoint específico
python train.py --config configs/train_config.yaml --pipeline unet_resnet50 --checkpoint-retomada checkpoints/unet_resnet50_last.pth
```

Ao retomar, são restaurados os pesos do modelo, o otimizador, o scheduler,
mixed precision, early stopping e o histórico de métricas. A retomada ocorre
na época seguinte à última época concluída e só é possível a partir de um
checkpoint salvo ao final de uma época.

O dispositivo (CPU / CUDA / MPS) é detectado automaticamente
(`utils/seed.py::detectar_dispositivo`).

### 5.5. Encontrar o limiar ideal e gerar os relatórios de avaliação

Os comandos abaixo devem ser executados a partir da pasta
`wound_segmentation`, depois que os checkpoints `_best.pth` tiverem sido
gerados. A avaliação usa as imagens de `data/validacao/`.

#### 1. Encontrar o limiar ideal

Execute a busca para cada arquitetura. O critério recomendado é `dice`;
também são aceitos `iou`, `acuracia` e `f1`.

```powershell
python evaluate_segmentation.py `
  --weights checkpoints/unet_best.pth `
  --architecture unet `
  --output_dir results/unet_threshold `
  --find_best_threshold `
  --threshold_start 0.10 --threshold_end 0.90 --threshold_step 0.05 `
  --threshold_metric dice

python evaluate_segmentation.py `
  --weights checkpoints/unet_resnet50_best.pth `
  --architecture unet_resnet50 `
  --output_dir results/unet_resnet50_threshold `
  --find_best_threshold `
  --threshold_start 0.10 --threshold_end 0.90 --threshold_step 0.05 `
  --threshold_metric dice
```

Como os diretórios de validação são os padrões, não é necessário informar
`--images_dir` e `--masks_dir`. Para conferir ou substituir os caminhos:

```powershell
--images_dir data/validacao/imagens --masks_dir data/validacao/mascaras
```

Cada busca grava o limiar selecionado em
`results/<modelo>_threshold/best_threshold.json` e todos os resultados da
varredura em `threshold_search/threshold_results.csv`.

#### 2. Avaliar usando o limiar encontrado

Leia o valor salvo em `best_threshold.json` e use-o na avaliação final.
O exemplo abaixo faz isso automaticamente no PowerShell:

```powershell
$limiar = (Get-Content results/unet_threshold/best_threshold.json | ConvertFrom-Json).threshold
python evaluate_segmentation.py `
  --weights checkpoints/unet_best.pth `
  --architecture unet `
  --output_dir results/unet_evaluation `
  --threshold $limiar `
  --batch_size 4

$limiar = (Get-Content results/unet_resnet50_threshold/best_threshold.json | ConvertFrom-Json).threshold
python evaluate_segmentation.py `
  --weights checkpoints/unet_resnet50_best.pth `
  --architecture unet_resnet50 `
  --output_dir results/unet_resnet50_evaluation `
  --threshold $limiar `
  --batch_size 4
```

Cada avaliação gera:

- `segmentation_results.csv`, com Dice, IoU, acurácia, precisão, recall,
  especificidade, F1, áreas e uma linha de resumo;
- `predicted_masks/`, com as máscaras binarizadas;
- `overlays/`, com a sobreposição entre máscara manual e predição;
- `evaluation.log`, com erros de imagens ou máscaras inválidas.

#### 3. Comparar todas as pipelines

Depois de avaliar individualmente os modelos, execute:

```bash
python3 evaluate.py --config configs/train_config.yaml
```

Esse comando avalia todos os checkpoints ativos em `data/validacao/` usando o
limiar definido em `avaliacao.limiar_binarizacao` no arquivo YAML e gera, em
`results/`:
- `metricas_por_amostra.csv` e `metricas_finais.json` por pipeline;
- `matriz_confusao.png` e comparações visuais por pipeline;
- `relatorio_comparativo_final.csv` / `.json` com as quatro pipelines lado a lado;
- `comparacao_dice_final.png` e `comparacao_iou_final.png`.

### 5.6. Inferência em uma nova imagem

```bash
python inference/predict.py \
    --imagem caminho/para/nova_imagem.png \
    --checkpoint checkpoints/unet_resnet50_best.pth \
    --arquitetura unet_resnet50 \
    --saida mascara_predita.png
```

### 5.7. Acompanhar o treinamento em tempo real

```bash
tensorboard --logdir logs/tensorboard
```

---

## 6. Explicação Técnica dos Componentes Principais

- **`models/unet.py`**: implementação própria da U-Net clássica (encoder e
  decoder construídos do zero, sem qualquer peso pré-treinado).
- **`models/unet_resnet50.py`**: U-Net cujo encoder é a ResNet-50 da
  torchvision (pesos ImageNet); o decoder é implementação própria, com skip
  connections extraídas dos quatro estágios da ResNet. O parâmetro
  `fine_tuning` controla se o encoder permanece treinável.
- **`preprocessing/opencv_pipeline.py`**: cada técnica (CLAHE, equalização,
  filtros de ruído, correção de iluminação, etc.) é um método isolado,
  ativado/desativado via YAML — usado exclusivamente pelas Pipelines 3 e 4.
- **`datasets/wound_dataset.py`**: a função `listar_amostras_com_mascara`
  varre o diretório configurado, casa cada imagem com a máscara de mesmo
  nome-base e descarta qualquer imagem sem par — é o ponto
  que garante que somente dados com máscara entrem no treino.
- **`datasets/augmentations.py`**: constrói o `A.Compose` do Albumentations
  dinamicamente, a partir da seção `data_augmentation` do YAML.
- **`training/trainer.py`**: loop de treino/validação com mixed precision
  (`torch.cuda.amp`), gradient clipping, `ReduceLROnPlateau`/cosine/step
  scheduler, early stopping configurável e checkpointing (`_best`/`_last`).
- **`training/losses.py`** e **`training/metrics.py`**: implementam todas as
  funções de perda e métricas exigidas, operando diretamente sobre logits.
- **`evaluation/evaluator.py`**: avalia no conjunto de validação, persiste
  métricas (CSV/JSON), gera matriz de confusão por pixel e comparações
  visuais; `gerar_relatorio_comparativo` consolida as quatro pipelines.

---

## 7. Relatório Comparativo entre as Quatro Abordagens

O relatório final (`results/relatorio_comparativo_final.csv`) é gerado
automaticamente após `evaluate.py` e contém, para cada pipeline, os valores
médios de Dice, IoU, Precisão, Recall, F1 e Acurácia no conjunto de validação.
De forma geral, o comportamento esperado — a ser confirmado empiricamente
com o dataset real — é:

- **U-Net + ResNet-50** tende a convergir mais rápido e alcançar Dice/IoU
  mais altos devido ao transfer learning, especialmente com datasets
  pequenos ou moderados.
- **U-Net pura** costuma exigir mais épocas e mais dados para atingir
  desempenho comparável, mas evita qualquer viés herdado do ImageNet.
- **A etapa OpenCV (Pipelines 3 e 4)** tende a ajudar mais em imagens com
  iluminação inconsistente ou baixo contraste entre ferida e pele — mas pode
  não trazer ganho (ou até prejudicar levemente) quando o dataset já é bem
  padronizado, pois pode remover informação de textura relevante.

A pipeline vencedora deve ser escolhida com base no relatório gerado sobre o
dataset real, não presumida a priori.

---

## 8. Reprodutibilidade

Todas as pipelines fixam seed global (`utils/seed.py::fixar_seed_global`) e
salvam, junto a cada experimento, a configuração completa utilizada
(`experiments/<pipeline>/configuracao_utilizada.json`), garantindo que os
resultados possam ser reproduzidos.
# Experimentos ResNet50 com partições fixas

Prepare as partições uma vez com `python preparar_particoes_resnet50.py`. A matriz executa com `python run_experiments.py` e lê somente `data/particoes/grid_resnet50` (imagens e máscaras); não acessa as pastas originais. Para recriar a divisão, use `python preparar_particoes_resnet50.py --recriar`.

Cada experimento salva configuração, checkpoints, logs, avaliação, threshold calibrado na validação e overlays dos 20 piores casos do teste em sua própria pasta. O ranking usa somente Dice no teste. O processo reduz o batch físico após falhas, elevando a acumulação para preservar o batch efetivo; se falhar também com batch 1, registra `skipped` e continua.

