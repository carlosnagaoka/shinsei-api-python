"""
Detector de Anomalias para Cobranças - Shinsei Logistics
Detecta valores suspeitos em cargas antes de gerar faturas
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Any

class DetectorAnomalias:
    """
    Classe para detectar valores anômalos em cargas de transporte
    """
    
    def __init__(self, contaminacao=0.1):
        """
        Args:
            contaminacao: Percentual esperado de anomalias (padrão: 10%)
        """
        self.modelo = IsolationForest(
            contamination=contaminacao,
            random_state=42,
            n_estimators=100
        )
        
    def analisar_cargas(self, cargas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analisa lista de cargas e identifica anomalias
        
        Args:
            cargas: Lista de dicionários com dados das cargas
            
        Returns:
            Dicionário com resultados da análise
        """
        if not cargas or len(cargas) < 3:
            return {
                'anomalias': [],
                'estatisticas': None,
                'aviso': 'Número insuficiente de cargas para análise (mínimo 3)'
            }
        
        # Extrair valores e converter para array
        valores = np.array([float(c.get('CARGAS_VALOR', 0)) for c in cargas])
        valores = valores.reshape(-1, 1)
        
        # Calcular estatísticas básicas
        estatisticas = self._calcular_estatisticas(valores.flatten())
        
        # Detectar anomalias usando Isolation Forest
        predicoes = self.modelo.fit_predict(valores)
        scores = self.modelo.score_samples(valores)
        
        # Identificar cargas anômalas
        anomalias = []
        for i, (predicao, score) in enumerate(zip(predicoes, scores)):
            if predicao == -1:  # -1 indica anomalia
                carga = cargas[i]
                valor = float(carga.get('CARGAS_VALOR', 0))
                
                # Determinar tipo de anomalia
                tipo_anomalia = self._classificar_anomalia(valor, estatisticas)
                
                # Calcular severidade (0-100)
                severidade = self._calcular_severidade(score, scores)
                
                anomalias.append({
                    'id': carga.get('ID_CARGAS'),
                    'veiculo': carga.get('CARGAS_VEICULO'),
                    'chassis': carga.get('CARGAS_CHASSIS'),
                    'valor': valor,
                    'tipo': tipo_anomalia,
                    'severidade': severidade,
                    'score_anomalia': float(score),
                    'sugestao': self._gerar_sugestao(valor, estatisticas, tipo_anomalia),
                    'valor_esperado': estatisticas['mediana']
                })
        
        return {
            'anomalias': anomalias,
            'estatisticas': estatisticas,
            'total_cargas': len(cargas),
            'total_anomalias': len(anomalias),
            'percentual_anomalias': round((len(anomalias) / len(cargas)) * 100, 2)
        }
    
    def _calcular_estatisticas(self, valores: np.ndarray) -> Dict[str, float]:
        """Calcula estatísticas descritivas dos valores"""
        return {
            'media': float(np.mean(valores)),
            'mediana': float(np.median(valores)),
            'desvio_padrao': float(np.std(valores)),
            'minimo': float(np.min(valores)),
            'maximo': float(np.max(valores)),
            'q1': float(np.percentile(valores, 25)),
            'q3': float(np.percentile(valores, 75)),
            'iqr': float(np.percentile(valores, 75) - np.percentile(valores, 25))
        }
    
    def _classificar_anomalia(self, valor: float, stats: Dict[str, float]) -> str:
        """Classifica o tipo de anomalia baseado nas estatísticas"""
        if valor > stats['q3'] + (1.5 * stats['iqr']):
            if valor > stats['media'] * 2:
                return 'MUITO_ALTO'
            return 'ALTO'
        elif valor < stats['q1'] - (1.5 * stats['iqr']):
            if valor < stats['media'] * 0.5:
                return 'MUITO_BAIXO'
            return 'BAIXO'
        return 'OUTLIER'
    
    def _calcular_severidade(self, score: float, todos_scores: np.ndarray) -> int:
        """
        Calcula severidade de 0-100 (100 = mais suspeito)
        Baseado em quão isolado o valor está
        """
        # Normalizar score para 0-100
        min_score = np.min(todos_scores)
        max_score = np.max(todos_scores)
        
        if max_score == min_score:
            return 50
        
        # Inverter: scores mais negativos = mais anômalos = maior severidade
        severidade = 100 - ((score - min_score) / (max_score - min_score) * 100)
        return int(severidade)
    
    def _gerar_sugestao(self, valor: float, stats: Dict[str, float], tipo: str) -> str:
        """Gera sugestão de ação baseada no tipo de anomalia"""
        sugestoes = {
            'MUITO_ALTO': f"Valor {valor:,.0f} está muito acima da média ({stats['media']:,.0f}). Verifique se não há zeros extras.",
            'ALTO': f"Valor {valor:,.0f} está acima do esperado ({stats['mediana']:,.0f}). Confirme se está correto.",
            'MUITO_BAIXO': f"Valor {valor:,.0f} está muito abaixo da média ({stats['media']:,.0f}). Verifique se não faltam dígitos.",
            'BAIXO': f"Valor {valor:,.0f} está abaixo do esperado ({stats['mediana']:,.0f}). Confirme se está correto.",
            'OUTLIER': f"Valor {valor:,.0f} é incomum. Valor típico é ¥{stats['mediana']:,.0f}."
        }
        return sugestoes.get(tipo, "Verifique este valor.")


class DetectorAnomaliasPorCliente:
    """
    Detector que considera histórico do cliente
    """
    
    def __init__(self):
        self.detector_base = DetectorAnomalias()
    
    def analisar_com_historico(
        self, 
        cargas_atuais: List[Dict[str, Any]], 
        historico_cliente: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analisa cargas considerando histórico do cliente
        """
        # Análise básica das cargas atuais
        resultado_atual = self.detector_base.analisar_cargas(cargas_atuais)
        
        # Se tiver histórico, fazer comparação adicional
        if historico_cliente and len(historico_cliente) >= 10:
            valores_historicos = [float(h.get('CARGAS_VALOR', 0)) for h in historico_cliente]
            stats_historico = {
                'media_historica': np.mean(valores_historicos),
                'mediana_historica': np.median(valores_historicos),
                'max_historico': np.max(valores_historicos),
                'min_historico': np.min(valores_historicos)
            }
            
            # Verificar se valores atuais estão muito fora do histórico
            for carga in cargas_atuais:
                valor = float(carga.get('CARGAS_VALOR', 0))
                
                # Se valor atual é 2x maior que máximo histórico
                if valor > stats_historico['max_historico'] * 2:
                    # Adicionar como anomalia se ainda não foi detectado
                    ja_detectado = any(
                        a['id'] == carga.get('ID_CARGAS') 
                        for a in resultado_atual['anomalias']
                    )
                    
                    if not ja_detectado:
                        resultado_atual['anomalias'].append({
                            'id': carga.get('ID_CARGAS'),
                            'veiculo': carga.get('CARGAS_VEICULO'),
                            'valor': valor,
                            'tipo': 'FORA_DO_HISTORICO',
                            'severidade': 90,
                            'sugestao': f"Valor ¥{valor:,.0f} é 2x maior que máximo histórico (¥{stats_historico['max_historico']:,.0f})",
                            'valor_esperado': stats_historico['media_historica']
                        })
            
            resultado_atual['comparacao_historico'] = stats_historico
        
        return resultado_atual