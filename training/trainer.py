"""
Módulo Trainer.

Implementa a classe `Trainer`, responsável por orquestrar o loop completo de
treinamento e validação: mixed precision, gradient clipping, scheduler de
learning rate, early stopping, checkpointing e logging via TensorBoard.
Utilizada por todas as quatro pipelines do projeto.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from training.metrics import calcular_metricas
from utils.logger import criar_logger


@dataclass
class HistoricoTreinamento:
    """Armazena o histórico de métricas ao longo das épocas, para plotagem posterior."""

    perda_treino: List[float] = field(default_factory=list)
    perda_validacao: List[float] = field(default_factory=list)
    dice_treino: List[float] = field(default_factory=list)
    dice_validacao: List[float] = field(default_factory=list)
    iou_treino: List[float] = field(default_factory=list)
    iou_validacao: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)


class EarlyStopping:
    """Interrompe o treinamento quando a métrica monitorada para de melhorar."""

    def __init__(self, paciencia: int = 15, modo: str = "max", delta_minimo: float = 1e-4) -> None:
        self.paciencia = paciencia
        self.modo = modo
        self.delta_minimo = delta_minimo
        self.contador = 0
        self.melhor_valor: Optional[float] = None
        self.deve_parar = False

    def atualizar(self, valor_atual: float) -> bool:
        """Atualiza o estado do early stopping com o valor mais recente da métrica.

        Returns:
            True se houve melhora (novo melhor valor), False caso contrário.
        """
        if self.melhor_valor is None:
            self.melhor_valor = valor_atual
            return True

        melhorou = (
            valor_atual > self.melhor_valor + self.delta_minimo
            if self.modo == "max"
            else valor_atual < self.melhor_valor - self.delta_minimo
        )

        if melhorou:
            self.melhor_valor = valor_atual
            self.contador = 0
            return True

        self.contador += 1
        if self.contador >= self.paciencia:
            self.deve_parar = True
        return False


class Trainer:
    """Orquestra o treinamento e a validação de um modelo de segmentação.

    Args:
        modelo: Rede neural a ser treinada.
        funcao_perda: Função de perda configurada (ver `training.losses`).
        dispositivo: Dispositivo de processamento (`torch.device`).
        config_treinamento: Dicionário com a seção `treinamento` do YAML.
        nome_experimento: Identificador da pipeline (usado em logs/checkpoints).
        diretorio_checkpoints: Diretório onde os `.pth` serão salvos.
        diretorio_logs: Diretório onde os logs (TensorBoard + texto) serão salvos.
    """

    def __init__(
        self,
        modelo: nn.Module,
        funcao_perda: nn.Module,
        dispositivo: torch.device,
        config_treinamento: Dict[str, Any],
        nome_experimento: str,
        diretorio_checkpoints: str = "checkpoints",
        diretorio_logs: str = "logs",
    ) -> None:
        self.modelo = modelo.to(dispositivo)
        self.funcao_perda = funcao_perda
        self.dispositivo = dispositivo
        self.config = config_treinamento
        self.nome_experimento = nome_experimento

        self.diretorio_checkpoints = Path(diretorio_checkpoints)
        self.diretorio_checkpoints.mkdir(parents=True, exist_ok=True)

        self.logger = criar_logger(f"trainer.{nome_experimento}", diretorio_logs)
        if SummaryWriter is None:
            self.logger.warning(
                "TensorBoard não está instalado no interpretador atual; os logs do TensorBoard serão desativados."
            )
            self.tensorboard = _NoOpSummaryWriter()
        else:
            self.tensorboard = SummaryWriter(log_dir=str(Path(diretorio_logs) / "tensorboard" / nome_experimento))

        self.otimizador = self._construir_otimizador()
        self.scheduler = self._construir_scheduler()

        self.usar_mixed_precision = bool(self.config.get("mixed_precision", True)) and dispositivo.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.usar_mixed_precision)

        config_clip = self.config.get("gradient_clipping", {})
        self.gradient_clipping_ativo = bool(config_clip.get("ativo", True))
        self.gradient_clip_max_norm = float(config_clip.get("max_norm", 1.0))

        config_early_stopping = self.config.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            paciencia=int(config_early_stopping.get("paciencia", 15)),
            modo=str(config_early_stopping.get("modo", "max")),
        )

        self.historico = HistoricoTreinamento()

    def _construir_otimizador(self) -> torch.optim.Optimizer:
        """Constrói o otimizador de acordo com a configuração ('adam', 'adamw' ou 'sgd')."""
        nome_otimizador = str(self.config.get("optimizer", "adamw")).lower()
        lr = float(self.config.get("learning_rate", 1e-4))
        weight_decay = float(self.config.get("weight_decay", 1e-5))

        parametros_treinaveis = filter(lambda p: p.requires_grad, self.modelo.parameters())

        if nome_otimizador == "adam":
            return torch.optim.Adam(parametros_treinaveis, lr=lr, weight_decay=weight_decay)
        if nome_otimizador == "sgd":
            return torch.optim.SGD(parametros_treinaveis, lr=lr, momentum=0.9, weight_decay=weight_decay)
        return torch.optim.AdamW(parametros_treinaveis, lr=lr, weight_decay=weight_decay)

    def _construir_scheduler(self):
        """Constrói o scheduler de learning rate conforme configurado."""
        config_scheduler = self.config.get("lr_scheduler", {})
        tipo = str(config_scheduler.get("tipo", "reduce_on_plateau")).lower()

        if tipo == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.otimizador, T_max=int(self.config.get("epochs", 100))
            )
        if tipo == "step":
            return torch.optim.lr_scheduler.StepLR(self.otimizador, step_size=20, gamma=0.5)

        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.otimizador,
            mode="max",
            patience=int(config_scheduler.get("paciencia", 5)),
            factor=float(config_scheduler.get("fator", 0.5)),
        )

    def treinar(self, loader_treino: DataLoader, loader_validacao: DataLoader) -> HistoricoTreinamento:
        """Executa o loop completo de treinamento até o número de épocas ou early stopping.

        Args:
            loader_treino: DataLoader do conjunto de treino.
            loader_validacao: DataLoader do conjunto de validação.

        Returns:
            Histórico completo de métricas de treino e validação.
        """
        epochs = int(self.config.get("epochs", 100))
        self.logger.info(f"Iniciando treinamento da pipeline '{self.nome_experimento}' por até {epochs} épocas.")

        for epoca in range(1, epochs + 1):
            inicio = time.time()

            perda_treino, dice_treino, iou_treino = self._executar_epoca(loader_treino, treinando=True)
            perda_validacao, dice_validacao, iou_validacao = self._executar_epoca(loader_validacao, treinando=False)

            lr_atual = self.otimizador.param_groups[0]["lr"]

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(dice_validacao)
            else:
                self.scheduler.step()

            self._registrar_epoca(
                epoca, perda_treino, perda_validacao, dice_treino, dice_validacao, iou_treino, iou_validacao, lr_atual
            )

            houve_melhora = self.early_stopping.atualizar(dice_validacao)
            self._salvar_checkpoints(epoca, dice_validacao, eh_melhor=houve_melhora)

            duracao = time.time() - inicio
            self.logger.info(
                f"Época {epoca}/{epochs} | "
                f"loss_treino={perda_treino:.4f} loss_val={perda_validacao:.4f} | "
                f"dice_treino={dice_treino:.4f} dice_val={dice_validacao:.4f} | "
                f"iou_val={iou_validacao:.4f} | lr={lr_atual:.2e} | tempo={duracao:.1f}s"
            )

            if self.config.get("early_stopping", {}).get("ativo", True) and self.early_stopping.deve_parar:
                self.logger.info(f"Early stopping acionado na época {epoca}. Encerrando treinamento.")
                break

        self.tensorboard.close()
        return self.historico

    def _executar_epoca(self, loader: DataLoader, treinando: bool) -> tuple[float, float, float]:
        """Executa uma época completa de treino ou validação."""
        self.modelo.train(mode=treinando)

        perda_acumulada = 0.0
        dice_acumulado = 0.0
        iou_acumulado = 0.0
        numero_batches = max(len(loader), 1)

        descricao = "Treino" if treinando else "Validação"
        barra_progresso = tqdm(loader, desc=descricao, leave=False)

        for imagens, mascaras in barra_progresso:
            imagens = imagens.to(self.dispositivo, non_blocking=True)
            mascaras = mascaras.to(self.dispositivo, non_blocking=True)

            with torch.set_grad_enabled(treinando):
                with torch.cuda.amp.autocast(enabled=self.usar_mixed_precision):
                    logits = self.modelo(imagens)
                    perda = self.funcao_perda(logits, mascaras)

                if treinando:
                    self.otimizador.zero_grad(set_to_none=True)
                    self.scaler.scale(perda).backward()

                    if self.gradient_clipping_ativo:
                        self.scaler.unscale_(self.otimizador)
                        torch.nn.utils.clip_grad_norm_(self.modelo.parameters(), self.gradient_clip_max_norm)

                    self.scaler.step(self.otimizador)
                    self.scaler.update()

            metricas = calcular_metricas(logits.detach(), mascaras)
            perda_acumulada += perda.item()
            dice_acumulado += metricas.dice
            iou_acumulado += metricas.iou
            barra_progresso.set_postfix(loss=perda.item(), dice=metricas.dice)

        return perda_acumulada / numero_batches, dice_acumulado / numero_batches, iou_acumulado / numero_batches

    def _registrar_epoca(
        self,
        epoca: int,
        perda_treino: float,
        perda_validacao: float,
        dice_treino: float,
        dice_validacao: float,
        iou_treino: float,
        iou_validacao: float,
        lr_atual: float,
    ) -> None:
        """Atualiza o histórico interno e registra as métricas no TensorBoard."""
        self.historico.perda_treino.append(perda_treino)
        self.historico.perda_validacao.append(perda_validacao)
        self.historico.dice_treino.append(dice_treino)
        self.historico.dice_validacao.append(dice_validacao)
        self.historico.iou_treino.append(iou_treino)
        self.historico.iou_validacao.append(iou_validacao)
        self.historico.learning_rates.append(lr_atual)

        self.tensorboard.add_scalars("Loss", {"treino": perda_treino, "validacao": perda_validacao}, epoca)
        self.tensorboard.add_scalars("Dice", {"treino": dice_treino, "validacao": dice_validacao}, epoca)
        self.tensorboard.add_scalars("IoU", {"treino": iou_treino, "validacao": iou_validacao}, epoca)
        self.tensorboard.add_scalar("Learning_Rate", lr_atual, epoca)

    def _salvar_checkpoints(self, epoca: int, dice_validacao: float, eh_melhor: bool) -> None:
        """Salva os checkpoints `_last.pth` e, se aplicável, `_best.pth`."""
        estado = {
            "epoca": epoca,
            "modelo_state_dict": self.modelo.state_dict(),
            "otimizador_state_dict": self.otimizador.state_dict(),
            "dice_validacao": dice_validacao,
        }

        config_checkpoint = self.config.get("checkpoint", {})

        if config_checkpoint.get("salvar_ultimo", True):
            torch.save(estado, self.diretorio_checkpoints / f"{self.nome_experimento}_last.pth")

        if config_checkpoint.get("salvar_melhor", True) and eh_melhor:
            torch.save(estado, self.diretorio_checkpoints / f"{self.nome_experimento}_best.pth")

    def salvar_configuracao_experimento(self, config_completa: Dict[str, Any], diretorio_saida: str) -> None:
        """Salva a configuração e os hiperparâmetros utilizados, para reprodutibilidade."""
        caminho = Path(diretorio_saida)
        caminho.mkdir(parents=True, exist_ok=True)
        with open(caminho / "configuracao_utilizada.json", "w", encoding="utf-8") as arquivo:
            json.dump(config_completa, arquivo, indent=2, ensure_ascii=False, default=str)
