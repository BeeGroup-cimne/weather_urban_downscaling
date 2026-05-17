#!/usr/bin/env python3
"""Sistema robusto de alertas de olas de calor con detección de anomalías y notificaciones resilientes."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import urllib.request
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy import ndimage

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("alerts.log")
    ]
)

logger = logging.getLogger("robust_alerts")

# Constantes
LEVEL_RANK = {
    "normal": 0,
    "watch": 1,
    "warning": 2,
    "severe": 3,
}

# Variables globales
STOP_REQUESTED = False


def _handle_signal(signum, frame) -> None:
    """Manejador de señales para detener el proceso de forma limpia."""
    del signum, frame
    global STOP_REQUESTED
    logger.info("Señal de detención recibida. Finalizando de forma ordenada...")
    STOP_REQUESTED = True


class DataValidator:
    """Validador de datos de predicción para detectar anomalías."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializar validador con configuración opcional.
        
        Args:
            config: Configuración personalizada para validación
        """
        # Configuración por defecto
        self.config = {
            "temp_min_valid": -20.0,  # Temperatura mínima válida (°C)
            "temp_max_valid": 60.0,   # Temperatura máxima válida (°C)
            "max_temporal_jump": 15.0,  # Salto máximo entre frames consecutivos (°C)
            "max_nan_fraction": 0.01,   # Fracción máxima de NaNs permitida
            "repair_nans": True,        # Intentar reparar NaNs
        }
        
        # Actualizar con configuración personalizada
        if config:
            self.config.update(config)
    
    def validate(self, prediction_array: np.ndarray, metadata: Optional[Dict] = None) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Validar datos de predicción para detectar anomalías.
        
        Args:
            prediction_array: Array de predicción
            metadata: Metadatos opcionales con información adicional
            
        Returns:
            Tuple con array (posiblemente reparado) y diccionario de advertencias (o None)
        """
        # Verificar forma y tipo
        if not isinstance(prediction_array, np.ndarray):
            raise ValueError(f"Predicción debe ser numpy array, no {type(prediction_array)}")
        
        if prediction_array.ndim not in (2, 3):
            raise ValueError(f"Predicción debe ser 2D o 3D, forma actual: {prediction_array.shape}")
        
        # Verificar valores NaN o infinitos
        if not np.isfinite(prediction_array).all():
            nan_count = np.isnan(prediction_array).sum()
            inf_count = np.isinf(prediction_array).sum()
            nan_fraction = (nan_count + inf_count) / prediction_array.size
            
            # Si hay pocos NaN y está habilitada la reparación, intentar repararlos
            if nan_fraction < self.config["max_nan_fraction"] and self.config["repair_nans"]:
                repaired_array = self._repair_nan_values(prediction_array)
                return repaired_array, {
                    "warning": "fixed_nans",
                    "nan_count": int(nan_count),
                    "inf_count": int(inf_count),
                    "fraction": float(nan_fraction)
                }
            else:
                raise ValueError(
                    f"Predicción contiene demasiados valores no válidos: "
                    f"{nan_count} NaNs, {inf_count} Infs ({nan_fraction:.2%})"
                )
        
        # Verificar rango físicamente plausible
        min_val = np.nanmin(prediction_array)
        max_val = np.nanmax(prediction_array)
        
        if min_val < self.config["temp_min_valid"] or max_val > self.config["temp_max_valid"]:
            return prediction_array, {
                "warning": "unusual_range",
                "min": float(min_val),
                "max": float(max_val),
                "expected_range": [self.config["temp_min_valid"], self.config["temp_max_valid"]]
            }
        
        # Verificar cambios bruscos (posible inestabilidad numérica)
        if prediction_array.ndim == 3 and prediction_array.shape[0] > 1:
            diffs = np.abs(np.diff(prediction_array, axis=0))
            max_diff = float(np.nanmax(diffs))
            if max_diff > self.config["max_temporal_jump"]:
                return prediction_array, {
                    "warning": "large_temporal_jump",
                    "max_diff": max_diff,
                    "threshold": self.config["max_temporal_jump"]
                }
        
        # Verificar coherencia con metadatos
        if metadata and "expected_range" in metadata:
            min_expected, max_expected = metadata["expected_range"]
            if min_val < min_expected - 10 or max_val > max_expected + 10:
                return prediction_array, {
                    "warning": "outside_expected_range",
                    "expected": [min_expected, max_expected],
                    "actual": [float(min_val), float(max_val)]
                }
        
        # Todo correcto
        return prediction_array, None
    
    def _repair_nan_values(self, array: np.ndarray) -> np.ndarray:
        """
        Reparar valores NaN o infinitos en un array.
        
        Args:
            array: Array con posibles valores NaN o infinitos
            
        Returns:
            Array reparado
        """
        # Crear una copia para no modificar el original
        result = array.copy()
        
        # Identificar valores no válidos
        invalid_mask = ~np.isfinite(result)
        
        if not invalid_mask.any():
            return result
        
        # Caso 2D
        if array.ndim == 2:
            # Usar interpolación para rellenar NaNs
            result = ndimage.gaussian_filter(np.nan_to_num(result), sigma=1)
            
            # Aplicar solo donde había valores inválidos
            array[invalid_mask] = result[invalid_mask]
            return array
        
        # Caso 3D (temporal)
        if array.ndim == 3:
            for t in range(array.shape[0]):
                frame = array[t]
                invalid_frame = invalid_mask[t]
                
                if invalid_frame.any():
                    # Intentar primero interpolación temporal
                    if t > 0 and t < array.shape[0] - 1:
                        # Promedio de frames anterior y siguiente
                        prev_frame = array[t-1]
                        next_frame = array[t+1]
                        
                        # Donde ambos frames son válidos
                        valid_interp = np.isfinite(prev_frame) & np.isfinite(next_frame)
                        interp_mask = invalid_frame & valid_interp
                        
                        if interp_mask.any():
                            array[t][interp_mask] = (prev_frame[interp_mask] + next_frame[interp_mask]) / 2
                    
                    # Para los restantes, usar interpolación espacial
                    remaining_invalid = ~np.isfinite(array[t])
                    if remaining_invalid.any():
                        fixed_frame = ndimage.gaussian_filter(np.nan_to_num(array[t]), sigma=1)
                        array[t][remaining_invalid] = fixed_frame[remaining_invalid]
        
        return array


class AnomalyDetector:
    """Detector de anomalías para predicciones de temperatura."""
    
    def __init__(self, history_file: Optional[str] = None, window_size: int = 24*7):
        """
        Inicializar detector de anomalías.
        
        Args:
            history_file: Archivo opcional con historial de predicciones
            window_size: Tamaño de la ventana de historial (en horas)
        """
        self.window_size = window_size
        self.history = []
        self.anomaly_log = []
        
        # Cargar historial si existe
        if history_file and Path(history_file).exists():
            try:
                df = pd.read_csv(history_file)
                self.history = df[["timestamp", "mean", "max", "min", "exceed_fraction"]].to_dict("records")
                logger.info(f"Historial cargado: {len(self.history)} registros de {history_file}")
            except Exception as e:
                logger.error(f"Error cargando historial: {e}")
    
    def update(self, timestamp: str, prediction_array: np.ndarray, threshold_temp: float = 30.0) -> Optional[Dict]:
        """
        Actualizar con nueva predicción y detectar anomalías.
        
        Args:
            timestamp: Timestamp de la predicción
            prediction_array: Array con datos de predicción
            threshold_temp: Temperatura umbral para calcular fracción excedida
            
        Returns:
            Diccionario con anomalías detectadas o None
        """
        # Calcular estadísticas
        stats = {
            "timestamp": timestamp,
            "mean": float(np.nanmean(prediction_array)),
            "max": float(np.nanmax(prediction_array)),
            "min": float(np.nanmin(prediction_array)),
            "exceed_fraction": float(np.mean(prediction_array > threshold_temp)),
            "processing_time": time.time()
        }
        
        # Añadir a historial
        self.history.append(stats)
        
        # Mantener ventana de historial
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]
        
        # Detectar anomalías si tenemos suficiente historial
        anomalies = {}
        if len(self.history) >= 24:  # Al menos un día de datos
            # Calcular estadísticas de la ventana
            recent = self.history[-24:]  # Últimas 24 horas
            means = [r["mean"] for r in recent]
            maxs = [r["max"] for r in recent]
            mins = [r["min"] for r in recent]
            
            # Detectar saltos bruscos (más de 5°C en una hora)
            if len(recent) >= 2:
                mean_diff = abs(stats["mean"] - recent[-2]["mean"])
                max_diff = abs(stats["max"] - recent[-2]["max"])
                
                if mean_diff > 5.0:
                    anomalies["mean_jump"] = mean_diff
                
                if max_diff > 8.0:
                    anomalies["max_jump"] = max_diff
            
            # Detectar valores atípicos (más de 3 desviaciones estándar)
            if len(means) >= 5:
                mean_std = np.std(means[:-1])  # Excluir el valor actual
                mean_avg = np.mean(means[:-1])
                
                z_score = abs(stats["mean"] - mean_avg) / (mean_std + 1e-6)
                if z_score > 3.0:
                    anomalies["mean_outlier"] = float(z_score)
                
                # Verificar tendencia sospechosa (aumento/disminución constante)
                diffs = np.diff(means[-6:])
                if (diffs > 0).all() and sum(diffs) > 10:
                    anomalies["rising_trend"] = float(sum(diffs))
                elif (diffs < 0).all() and sum(diffs) < -10:
                    anomalies["falling_trend"] = float(sum(diffs))
        
        # Registrar anomalías
        if anomalies:
            self.anomaly_log.append({
                "timestamp": timestamp,
                "anomalies": anomalies,
                "stats": stats
            })
            
            logger.warning(f"Anomalías detectadas: {anomalies}")
            return anomalies
        
        return None
    
    def save_history(self, file_path: str) -> None:
        """
        Guardar historial para uso futuro.
        
        Args:
            file_path: Ruta donde guardar el historial
        """
        # Crear directorio si no existe
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar como CSV
        pd.DataFrame(self.history).to_csv(file_path, index=False)
        logger.info(f"Historial guardado en {file_path}: {len(self.history)} registros")
    
    def get_anomaly_log(self) -> List[Dict]:
        """
        Obtener registro de anomalías.
        
        Returns:
            Lista de anomalías detectadas
        """
        return self.anomaly_log


class DynamicAlertThresholds:
    """Umbrales dinámicos para alertas basados en datos históricos."""
    
    def __init__(self, history_file: Optional[str] = None, seasonal_adjust: bool = True):
        """
        Inicializar sistema de umbrales dinámicos.
        
        Args:
            history_file: Archivo opcional con historial de predicciones
            seasonal_adjust: Si se deben ajustar umbrales por estacionalidad
        """
        # Umbrales base
        self.base_thresholds = {
            "normal": 28.0,   # Temperatura normal
            "watch": 30.0,    # Vigilancia
            "warning": 33.0,  # Alerta
            "severe": 36.0    # Alerta severa
        }
        
        self.seasonal_adjust = seasonal_adjust
        self.history_data = None
        
        # Cargar datos históricos si están disponibles
        if history_file and Path(history_file).exists():
            try:
                self.history_data = pd.read_csv(history_file)
                self.history_data["date"] = pd.to_datetime(self.history_data["timestamp"])
                self.history_data["month"] = self.history_data["date"].dt.month
                self.history_data["hour"] = self.history_data["date"].dt.hour
                logger.info(f"Datos históricos cargados: {len(self.history_data)} registros de {history_file}")
            except Exception as e:
                logger.error(f"Error cargando datos históricos: {e}")
    
    def get_thresholds(self, timestamp: Optional[Union[str, datetime]] = None) -> Dict[str, float]:
        """
        Obtener umbrales ajustados para el timestamp dado.
        
        Args:
            timestamp: Timestamp para el que se calculan los umbrales
            
        Returns:
            Diccionario con umbrales para cada nivel de alerta
        """
        # Usar umbrales base si no hay ajuste estacional o datos históricos
        if not self.seasonal_adjust or self.history_data is None or timestamp is None:
            return self.base_thresholds.copy()
        
        # Convertir timestamp a datetime si es string
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        # Extraer mes y hora
        month = timestamp.month
        hour = timestamp.hour
        
        # Filtrar datos históricos para el mismo mes y hora
        mask = (self.history_data["month"] == month) & (self.history_data["hour"] == hour)
        relevant_data = self.history_data[mask]
        
        # Si no hay suficientes datos, usar umbrales base
        if len(relevant_data) < 24:  # Al menos un día de datos
            logger.info(f"Datos históricos insuficientes para mes={month}, hora={hour}. Usando umbrales base.")
            return self.base_thresholds.copy()
        
        # Calcular percentiles para ajustar umbrales
        p75 = relevant_data["max"].quantile(0.75)
        p90 = relevant_data["max"].quantile(0.90)
        p95 = relevant_data["max"].quantile(0.95)
        p99 = relevant_data["max"].quantile(0.99)
        
        # Ajustar umbrales basados en percentiles históricos
        adjusted = {
            "normal": max(self.base_thresholds["normal"], p75 - 2),
            "watch": max(self.base_thresholds["watch"], p90 - 1),
            "warning": max(self.base_thresholds["warning"], p95),
            "severe": max(self.base_thresholds["severe"], p99)
        }
        
        logger.info(f"Umbrales ajustados para mes={month}, hora={hour}: {adjusted}")
        return adjusted
    
    def update_base_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """
        Actualizar umbrales base.
        
        Args:
            new_thresholds: Nuevos valores de umbrales
        """
        for level, value in new_thresholds.items():
            if level in self.base_thresholds:
                self.base_thresholds[level] = float(value)
        
        logger.info(f"Umbrales base actualizados: {self.base_thresholds}")


class EnhancedHeatwaveDetector:
    """Detector mejorado de eventos de ola de calor."""
    
    def __init__(self, min_duration_hours: int = 6, cooldown_hours: int = 12):
        """
        Inicializar detector de eventos de ola de calor.
        
        Args:
            min_duration_hours: Duración mínima para considerar un evento válido
            cooldown_hours: Horas sin alertas significativas para considerar finalizado un evento
        """
        self.min_duration_hours = min_duration_hours
        self.cooldown_hours = cooldown_hours
        self.current_event = None
        self.past_events = []
        self.alert_history = []
        
        logger.info(f"Detector de olas de calor inicializado: duración mínima={min_duration_hours}h, "
                   f"cooldown={cooldown_hours}h")
    
    def process_alert(self, timestamp: Union[str, datetime], alert_data: Dict) -> Dict:
        """
        Procesar una nueva alerta y actualizar eventos.
        
        Args:
            timestamp: Timestamp de la alerta
            alert_data: Datos de la alerta
            
        Returns:
            Diccionario con estado del evento
        """
        # Convertir timestamp a datetime si es string
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        # Registrar alerta en historial
        self.alert_history.append({
            "timestamp": timestamp,
            "level": alert_data.get("alert_level", "normal"),
            "max_temp": alert_data.get("max_pred_c"),
            "exceed_fraction": alert_data.get("exceed_fraction", 0)
        })
        
        # Mantener historial limitado
        if len(self.alert_history) > 168:  # 7 días
            self.alert_history = self.alert_history[-168:]
        
        # Determinar si es un nivel de alerta significativo
        is_significant = alert_data.get("alert_level") in ("warning", "severe")
        
        # Caso 1: No hay evento activo
        if self.current_event is None:
            if is_significant:
                # Iniciar nuevo evento
                self.current_event = {
                    "start": timestamp,
                    "last_update": timestamp,
                    "peak_level": alert_data.get("alert_level"),
                    "peak_temp": alert_data.get("max_pred_c"),
                    "peak_area": alert_data.get("exceed_fraction", 0),
                    "alerts": [alert_data]
                }
                logger.info(f"Nuevo evento de ola de calor iniciado: {timestamp}, nivel={alert_data.get('alert_level')}")
                return {"event_status": "started", "event": self.current_event}
        
        # Caso 2: Hay un evento activo
        else:
            # Actualizar último timestamp
            self.current_event["last_update"] = timestamp
            self.current_event["alerts"].append(alert_data)
            
            # Actualizar picos si corresponde
            if is_significant:
                current_level = alert_data.get("alert_level")
                current_temp = alert_data.get("max_pred_c")
                current_area = alert_data.get("exceed_fraction", 0)
                
                level_rank = {"normal": 0, "watch": 1, "warning": 2, "severe": 3}
                
                if level_rank.get(current_level, 0) > level_rank.get(self.current_event["peak_level"], 0):
                    logger.info(f"Evento actualizado: nivel {self.current_event['peak_level']} -> {current_level}")
                    self.current_event["peak_level"] = current_level
                
                if current_temp > self.current_event["peak_temp"]:
                    logger.info(f"Evento actualizado: temperatura pico {self.current_event['peak_temp']:.1f}°C -> {current_temp:.1f}°C")
                    self.current_event["peak_temp"] = current_temp
                
                if current_area > self.current_event["peak_area"]:
                    logger.info(f"Evento actualizado: área afectada {self.current_event['peak_area']:.1%} -> {current_area:.1%}")
                    self.current_event["peak_area"] = current_area
                
                # Reiniciar cooldown
                return {"event_status": "updated", "event": self.current_event}
            
            # Verificar si el evento ha terminado (periodo sin alertas significativas)
            significant_alerts = [a for a in self.alert_history 
                                if a.get("level") in ("warning", "severe")]
            
            if significant_alerts:
                last_significant = max(significant_alerts, key=lambda a: a["timestamp"])
                hours_since_peak = (timestamp - last_significant["timestamp"]).total_seconds() / 3600
            else:
                hours_since_peak = self.cooldown_hours + 1  # Forzar finalización
            
            if hours_since_peak >= self.cooldown_hours:
                # Finalizar evento actual
                self.current_event["end"] = timestamp
                self.current_event["duration_hours"] = (self.current_event["end"] - self.current_event["start"]).total_seconds() / 3600
                
                # Solo registrar eventos que cumplan duración mínima
                if self.current_event["duration_hours"] >= self.min_duration_hours:
                    self.past_events.append(self.current_event)
                    logger.info(f"Evento finalizado: duración={self.current_event['duration_hours']:.1f}h, "
                               f"nivel={self.current_event['peak_level']}, "
                               f"temp={self.current_event['peak_temp']:.1f}°C")
                    result = {"event_status": "ended", "event": self.current_event}
                    self.current_event = None
                    return result
                else:
                    # Descartar evento demasiado corto
                    logger.info(f"Evento descartado por duración insuficiente: {self.current_event['duration_hours']:.1f}h < {self.min_duration_hours}h")
                    result = {"event_status": "discarded", "event": self.current_event}
                    self.current_event = None
                    return result
        
        return {"event_status": "unchanged"}
    
    def get_active_event(self) -> Optional[Dict]:
        """
        Obtener evento activo si existe.
        
        Returns:
            Diccionario con evento activo o None
        """
        return self.current_event
    
    def get_past_events(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Obtener eventos pasados.
        
        Args:
            limit: Número máximo de eventos a devolver
            
        Returns:
            Lista de eventos pasados
        """
        if limit:
            return self.past_events[-limit:]
        return self.past_events
    
    def save_events(self, file_path: str) -> None:
        """
        Guardar eventos en archivo CSV.
        
        Args:
            file_path: Ruta donde guardar los eventos
        """
        # Preparar datos para CSV
        events_data = []
        for event in self.past_events:
            events_data.append({
                "start": event["start"].strftime("%Y-%m-%d %H:%M:%S"),
                "end": event["end"].strftime("%Y-%m-%d %H:%M:%S"),
                "duration_hours": event["duration_hours"],
                "peak_level": event["peak_level"],
                "peak_temp": event["peak_temp"],
                "peak_area": event["peak_area"]
            })
        
        # Guardar como CSV
        if events_data:
            df = pd.DataFrame(events_data)
            df.to_csv(file_path, index=False)
            logger.info(f"Eventos guardados en {file_path}: {len(events_data)} eventos")


class RobustNotificationSystem:
    """Sistema robusto de notificaciones con reintentos y confirmaciones."""
    
    def __init__(self, config: Optional[Union[Dict, str]] = None):
        """
        Inicializar sistema de notificaciones.
        
        Args:
            config: Configuración personalizada (diccionario o ruta a archivo JSON)
        """
        import json
        from pathlib import Path
        
        # Configuración por defecto
        self.config = {
            "webhooks": [],
            "email": {
                "enabled": False,
                "smtp_server": "",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "from_address": "",
                "recipients": []
            },
            "retry": {
                "max_attempts": 5,
                "initial_delay": 30,  # segundos
                "max_delay": 1800,    # 30 minutos
                "backoff_factor": 2
            },
            "notification_levels": {
                "normal": [],
                "watch": ["webhook"],
                "warning": ["webhook", "email"],
                "severe": ["webhook", "email"]
            }
        }
        
        # Cargar configuración personalizada
        if config:
            if isinstance(config, dict):
                self._update_config(config)
            elif isinstance(config, (str, Path)) and Path(config).exists():
                try:
                    with open(config, 'r') as f:
                        self._update_config(json.load(f))
                except Exception as e:
                    logger.error(f"Error cargando configuración: {e}")
        
        # Estado interno
        self.pending_notifications = []
        self.notification_history = []
        
        logger.info(f"Sistema de notificaciones inicializado: {len(self.config['webhooks'])} webhooks, "
                   f"email {'habilitado' if self.config['email']['enabled'] else 'deshabilitado'}")
    
    def _update_config(self, new_config: Dict) -> None:
        """
        Actualizar configuración de forma recursiva.
        
        Args:
            new_config: Nueva configuración
        """
        for key, value in new_config.items():
            if key in self.config:
                if isinstance(value, dict) and isinstance(self.config[key], dict):
                    self._update_config_dict(self.config[key], value)
                else:
                    self.config[key] = value
    
    def _update_config_dict(self, target: Dict, source: Dict) -> None:
        """
        Actualizar diccionario de configuración.
        
        Args:
            target: Diccionario destino
            source: Diccionario fuente
        """
        for key, value in source.items():
            if key in target and isinstance(value, dict) and isinstance(target[key], dict):
                self._update_config_dict(target[key], value)
            else:
                target[key] = value
    
    def notify(self, alert_data: Dict, channels: Optional[List[str]] = None) -> Dict:
        """
        Enviar notificación por los canales configurados.
        
        Args:
            alert_data: Datos de la alerta
            channels: Canales específicos a utilizar (opcional)
            
        Returns:
            Diccionario con estado de la notificación
        """
        import time
        import uuid
        
        # Determinar canales basados en nivel de alerta
        if channels is None:
            level = alert_data.get("alert_level", "normal")
            channels = self.config["notification_levels"].get(level, [])
        
        if not channels:
            return {"status": "skipped", "reason": "no_channels"}
        
        # Crear ID único para esta notificación
        notification_id = str(uuid.uuid4())
        
        # Preparar notificación
        notification = {
            "id": notification_id,
            "timestamp": time.time(),
            "alert_data": alert_data,
            "channels": channels,
            "attempts": 0,
            "status": "pending",
            "next_attempt": time.time(),
            "results": {}
        }
        
        # Añadir a cola pendiente
        self.pending_notifications.append(notification)
        
        # Procesar inmediatamente
        self._process_pending_notifications()
        
        return {"status": "queued", "notification_id": notification_id}
    
    def _process_pending_notifications(self) -> None:
        """Procesar notificaciones pendientes."""
        import time
        
        current_time = time.time()
        remaining = []
        
        for notification in self.pending_notifications:
            # Verificar si es momento de reintentar
            if notification["next_attempt"] <= current_time:
                # Incrementar contador de intentos
                notification["attempts"] += 1
                
                # Enviar por cada canal
