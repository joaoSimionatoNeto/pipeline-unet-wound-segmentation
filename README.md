# Wound Segmentation — Pipeline de Treinamento e Comparação de CNNs

Solução completa para treinamento, validação, avaliação e comparação de quatro
abordagens de segmentação semântica aplicadas a imagens de feridas.

---

## 1. Onde colocar as imagens (estrutura da pasta `data/`)

Esta é a parte mais importante para você começar a usar o projeto. A pasta
`data/` já vem separada com nomes explícitos, sem ambiguidade entre imagens
com e sem máscara:

```text
data/
│
├── com_mascara/
│   ├── imagens/               <-  imagens da ferida
│   └── mascaras/               ← mascara de segmenteção (mesmo nome de arquivo da imagem)
│
└── processed/                 ← gerado automaticamente pelo pipeline (cache)
                                  não é necessário colocar nada aqui manualmente
```

### Regra de pareamento

Uma imagem em `data/com_mascara/imagens/ferida_001.png` **precisa** ter uma
máscara com **exatamente o mesmo nome de arquivo** em
`data/com_mascara/mascaras/ferida_001.png`. O sistema (`datasets/wound_dataset.py`)
varre `imagens/`, procura o arquivo de mesmo nome em `mascaras/` e **ignora
automaticamente** qualquer imagem que não tenha máscara correspondente — por
isso não há problema em deixar imagens "soltas" dentro de `com_mascara/imagens/`,
mas o recomendado é manter apenas pares completos ali.

---

## 2. Visão Geral das Quatro Pipelines

| # | Nome interno | Pré-processamento OpenCV | Encoder | Transfer Learning |
|---|----------|--------------------|---------|--------------------|
| 1 | `unet_resnet50` | Não | ResNet-50 (ImageNet) | Sim |
| 2 | `unet` | Não | Construído do zero | Não |
| 3 | `unet_resnet50_opencv` | Sim | ResNet-50 (ImageNet) | Sim |
| 4 | `unet_opencv` | Sim | Construído do zero | Não |

Todas as pipelines leem os dados exclusivamente de `data/com_mascara/`.

---

## 3. Estrutura Completa do Repositório

```text
wound_segmentation/
│
├── data/                        # Seus dados (ver seção 1)
│   ├── com_mascara/
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
│   └── evaluator.py             # Avaliação no conjunto de teste + relatório comparativo
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

1. Você coloca os dados em `data/com_mascara/imagens` e `data/com_mascara/mascaras`.
2. `train.py` lê `configs/train_config.yaml`, monta os datasets
   (`datasets/`), aplica pré-processamento OpenCV se a pipeline exigir
   (`preprocessing/`), aplica augmentation e treina o modelo (`models/` +
   `training/`).
3. Durante o treino, `training/trainer.py` grava logs em `logs/`, salva
   checkpoints em `checkpoints/` e a configuração usada em `experiments/<pipeline>/`.
4. `evaluate.py` carrega os checkpoints `_best.pth`, avalia no conjunto de
   teste (`evaluation/`) e grava métricas e gráficos em `results/`.
5. `inference/predict.py` usa um checkpoint já treinado para gerar a máscara
   de uma imagem nova, fora do fluxo de treino.

---

## 4. Fluxograma da Solução

```text
                 ┌─────────────────────────────────────────┐
                 │   data/com_mascara/imagens + mascaras     │
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
            │   Evaluator (test set)   │
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

Siga a seção 1 deste README: imagens com máscara em `data/com_mascara/imagens/`
e `data/com_mascara/mascaras/` (mesmo nome de arquivo); imagens sem máscara
em `data/sem_mascara/imagens/`.

### 5.3. Ajustar a configuração

Edite `configs/train_config.yaml` para ativar/desativar pipelines, técnicas
de pré-processamento OpenCV, augmentations, hiperparâmetros de treino e a
função de perda. Os caminhos de dados já apontam para a nova estrutura
(`caminhos.imagens_com_mascara`, `caminhos.mascaras`, `caminhos.imagens_sem_mascara`).

### 5.4. Treinar

```bash
# Treinar todas as quatro pipelines sequencialmente
python train.py --config configs/train_config.yaml

# Treinar apenas uma pipeline específica
python train.py --config configs/train_config.yaml --pipeline unet_resnet50

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

### 5.5. Avaliar e comparar

```bash
python evaluate.py --config configs/train_config.yaml
```

Isso gera, em `results/`:
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
  varre `data/com_mascara/imagens/`, casa cada arquivo com o correspondente em
  `data/com_mascara/mascaras/` e descarta qualquer imagem sem par — é o ponto
  que garante que somente dados com máscara entrem no treino.
- **`datasets/augmentations.py`**: constrói o `A.Compose` do Albumentations
  dinamicamente, a partir da seção `data_augmentation` do YAML.
- **`training/trainer.py`**: loop de treino/validação com mixed precision
  (`torch.cuda.amp`), gradient clipping, `ReduceLROnPlateau`/cosine/step
  scheduler, early stopping configurável e checkpointing (`_best`/`_last`).
- **`training/losses.py`** e **`training/metrics.py`**: implementam todas as
  funções de perda e métricas exigidas, operando diretamente sobre logits.
- **`evaluation/evaluator.py`**: avalia no conjunto de teste, persiste
  métricas (CSV/JSON), gera matriz de confusão por pixel e comparações
  visuais; `gerar_relatorio_comparativo` consolida as quatro pipelines.

---

## 7. Relatório Comparativo entre as Quatro Abordagens

O relatório final (`results/relatorio_comparativo_final.csv`) é gerado
automaticamente após `evaluate.py` e contém, para cada pipeline, os valores
médios de Dice, IoU, Precisão, Recall, F1 e Acurácia no conjunto de teste.
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


