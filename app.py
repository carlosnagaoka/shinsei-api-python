# API Flask para gerenciamento de entregas e detecção de anomalias
# Autor: Shinsei API
# Versão: 2.0 - Com Detector de Anomalias

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializa a aplicação Flask
app = Flask(__name__)
CORS(app)  # Permite requisições do HostGator

# ============================================================
# CLASSE: DETECTOR DE ANOMALIAS
# ============================================================

class DetectorAnomalias:
    """
    Detector de valores anômalos em cargas de transporte.
    Usa algoritmo Isolation Forest para identificar valores suspeitos.
    """
    
    def __init__(self, contaminacao=0.15):
        """
        Args:
            contaminacao: Percentual esperado de anomalias (padrão: 15%)
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
        # Validação mínima
        if not cargas or len(cargas) < 3:
            return {
                'anomalias': [],
                'estatisticas': None,
                'aviso': 'Número insuficiente de cargas para análise (mínimo 3)'
            }
        
        try:
            # Extrair valores
            valores = np.array([float(c.get('CARGAS_VALOR', 0)) for c in cargas])
            valores = valores.reshape(-1, 1)
            
            # Calcular estatísticas
            estatisticas = self._calcular_estatisticas(valores.flatten())
            
            # Detectar anomalias
            predicoes = self.modelo.fit_predict(valores)
            scores = self.modelo.score_samples(valores)
            
            # Identificar cargas anômalas
            anomalias = []
            for i, (predicao, score) in enumerate(zip(predicoes, scores)):
                if predicao == -1:  # -1 = anomalia
                    carga = cargas[i]
                    valor = float(carga.get('CARGAS_VALOR', 0))
                    
                    tipo_anomalia = self._classificar_anomalia(valor, estatisticas)
                    severidade = self._calcular_severidade(score, scores)
                    
                    anomalias.append({
                        'id': carga.get('ID_CARGAS'),
                        'veiculo': carga.get('CARGAS_VEICULO', 'N/A'),
                        'chassis': carga.get('CARGAS_CHASSIS', 'N/A'),
                        'valor': valor,
                        'tipo': tipo_anomalia,
                        'severidade': severidade,
                        'score_anomalia': float(score),
                        'sugestao': self._gerar_sugestao(valor, estatisticas, tipo_anomalia),
                        'valor_esperado': int(estatisticas['mediana'])
                    })
            
            return {
                'anomalias': anomalias,
                'estatisticas': estatisticas,
                'total_cargas': len(cargas),
                'total_anomalias': len(anomalias),
                'percentual_anomalias': round((len(anomalias) / len(cargas)) * 100, 2)
            }
            
        except Exception as e:
            logger.error(f"Erro ao analisar cargas: {str(e)}")
            return {
                'anomalias': [],
                'estatisticas': None,
                'erro': str(e)
            }
    
    def _calcular_estatisticas(self, valores: np.ndarray) -> Dict[str, float]:
        """Calcula estatísticas descritivas dos valores"""
        return {
            'media': int(np.mean(valores)),
            'mediana': int(np.median(valores)),
            'desvio_padrao': int(np.std(valores)),
            'minimo': int(np.min(valores)),
            'maximo': int(np.max(valores)),
            'q1': int(np.percentile(valores, 25)),
            'q3': int(np.percentile(valores, 75)),
            'iqr': int(np.percentile(valores, 75) - np.percentile(valores, 25))
        }
    
    def _classificar_anomalia(self, valor: float, stats: Dict[str, float]) -> str:
        """Classifica o tipo de anomalia"""
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
        """Calcula severidade de 0-100 (100 = mais suspeito)"""
        min_score = np.min(todos_scores)
        max_score = np.max(todos_scores)
        
        if max_score == min_score:
            return 50
        
        # Inverter: scores mais negativos = maior severidade
        severidade = 100 - ((score - min_score) / (max_score - min_score) * 100)
        return int(severidade)
    
    def _gerar_sugestao(self, valor: float, stats: Dict[str, float], tipo: str) -> str:
        """Gera sugestão de ação"""
        sugestoes = {
            'MUITO_ALTO': f"Valor ¥{valor:,.0f} está muito acima da média (¥{stats['media']:,.0f}). Verifique se não há zeros extras.",
            'ALTO': f"Valor ¥{valor:,.0f} está acima do esperado (¥{stats['mediana']:,.0f}). Confirme se está correto.",
            'MUITO_BAIXO': f"Valor ¥{valor:,.0f} está muito abaixo da média (¥{stats['media']:,.0f}). Verifique se não faltam dígitos.",
            'BAIXO': f"Valor ¥{valor:,.0f} está abaixo do esperado (¥{stats['mediana']:,.0f}). Confirme se está correto.",
            'OUTLIER': f"Valor ¥{valor:,.0f} é incomum. Valor típico é ¥{stats['mediana']:,.0f}."
        }
        return sugestoes.get(tipo, "Verifique este valor.")


class DetectorAnomaliasPorCliente:
    """Detector que considera histórico do cliente"""
    
    def __init__(self):
        self.detector_base = DetectorAnomalias()
    
    def analisar_com_historico(
        self, 
        cargas_atuais: List[Dict[str, Any]], 
        historico_cliente: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analisa cargas considerando histórico do cliente"""
        # Análise básica
        resultado_atual = self.detector_base.analisar_cargas(cargas_atuais)
        
        # Se tiver histórico suficiente, fazer comparação adicional
        if historico_cliente and len(historico_cliente) >= 10:
            try:
                valores_historicos = [float(h.get('CARGAS_VALOR', 0)) for h in historico_cliente]
                stats_historico = {
                    'media_historica': int(np.mean(valores_historicos)),
                    'mediana_historica': int(np.median(valores_historicos)),
                    'max_historico': int(np.max(valores_historicos)),
                    'min_historico': int(np.min(valores_historicos))
                }
                
                # Verificar valores muito fora do histórico
                for carga in cargas_atuais:
                    valor = float(carga.get('CARGAS_VALOR', 0))
                    
                    # Se valor é 2x maior que máximo histórico
                    if valor > stats_historico['max_historico'] * 2:
                        ja_detectado = any(
                            a['id'] == carga.get('ID_CARGAS') 
                            for a in resultado_atual['anomalias']
                        )
                        
                        if not ja_detectado:
                            resultado_atual['anomalias'].append({
                                'id': carga.get('ID_CARGAS'),
                                'veiculo': carga.get('CARGAS_VEICULO', 'N/A'),
                                'valor': valor,
                                'tipo': 'FORA_DO_HISTORICO',
                                'severidade': 90,
                                'sugestao': f"Valor ¥{valor:,.0f} é 2x maior que máximo histórico (¥{stats_historico['max_historico']:,.0f})",
                                'valor_esperado': stats_historico['media_historica']
                            })
                
                resultado_atual['comparacao_historico'] = stats_historico
                
            except Exception as e:
                logger.error(f"Erro ao processar histórico: {str(e)}")
        
        return resultado_atual


# Instanciar detectores globalmente
detector = DetectorAnomalias(contaminacao=0.15)
detector_cliente = DetectorAnomaliasPorCliente()


# ============================================================
# ROTAS DA API
# ============================================================

@app.route('/', methods=['GET'])
def home():
    """Rota raiz - informações da API"""
    return jsonify({
        'nome': 'Shinsei Logistics API',
        'versao': '2.0.0',
        'status': 'online',
        'endpoints': {
            '/api/status': 'GET - Status da API',
            '/api/relatorio': 'POST - Gerar relatório de entregas',
            '/api/detectar-anomalias': 'POST - Detectar anomalias em cargas',
            '/api/detectar-anomalias-historico': 'POST - Detectar com histórico do cliente'
        }
    })


@app.route('/api/status', methods=['GET'])
def status():
    """
    Rota de verificação de status da API.
    Retorna uma mensagem confirmando que a API está funcionando.
    """
    return jsonify({
        'mensagem': 'API funcionando',
        'status': 'ok',
        'versao': '2.0.0'
    })


@app.route('/api/relatorio', methods=['POST'])
def relatorio():
    """
    Rota para gerar relatório de entregas.
    Recebe uma lista de entregas e retorna estatísticas.

    Exemplo de entrada:
    {
        "entregas": [
            {"id": 1, "valor": 100.0, "status": "entregue"},
            {"id": 2, "valor": 50.0, "status": "pendente"}
        ]
    }
    """
    # Obtém os dados JSON da requisição
    dados = request.get_json()

    # Verifica se os dados foram enviados corretamente
    if not dados or 'entregas' not in dados:
        return jsonify({
            'erro': 'Dados inválidos. Envie uma lista de entregas.'
        }), 400

    entregas = dados['entregas']

    # Calcula as estatísticas
    total_entregas = len(entregas)

    # Conta entregas por status
    entregues = sum(1 for e in entregas if e.get('status') == 'entregue')
    pendentes = sum(1 for e in entregas if e.get('status') == 'pendente')
    canceladas = sum(1 for e in entregas if e.get('status') == 'cancelado')

    # Calcula valor total das entregas
    valor_total = sum(e.get('valor', 0) for e in entregas)

    # Calcula valor das entregas concluídas
    valor_entregue = sum(
        e.get('valor', 0) for e in entregas
        if e.get('status') == 'entregue'
    )

    # Monta o relatório de estatísticas
    estatisticas = {
        'total_entregas': total_entregas,
        'entregues': entregues,
        'pendentes': pendentes,
        'canceladas': canceladas,
        'valor_total': round(valor_total, 2),
        'valor_entregue': round(valor_entregue, 2),
        'taxa_sucesso': round((entregues / total_entregas * 100), 2) if total_entregas > 0 else 0
    }

    return jsonify({
        'mensagem': 'Relatório gerado com sucesso',
        'estatisticas': estatisticas
    })


@app.route('/api/detectar-anomalias', methods=['POST'])
def detectar_anomalias():
    """
    Detecta anomalias em lista de cargas.
    
    Exemplo de entrada:
    {
        "cargas": [
            {
                "ID_CARGAS": 1,
                "CARGAS_VEICULO": "Toyota Corolla",
                "CARGAS_CHASSIS": "ABC123",
                "CARGAS_VALOR": 45000
            }
        ]
    }
    
    Retorna:
    {
        "sucesso": true,
        "resultado": {
            "anomalias": [...],
            "estatisticas": {...},
            "total_cargas": 10,
            "total_anomalias": 2
        }
    }
    """
    try:
        dados = request.get_json()
        
        if not dados or 'cargas' not in dados:
            return jsonify({
                'sucesso': False,
                'erro': 'Dados inválidos. Envie {"cargas": [...]}'
            }), 400
        
        cargas = dados['cargas']
        
        if not isinstance(cargas, list):
            return jsonify({
                'sucesso': False,
                'erro': 'Cargas deve ser uma lista'
            }), 400
        
        logger.info(f"Analisando {len(cargas)} cargas...")
        
        # Realizar análise
        resultado = detector.analisar_cargas(cargas)
        
        logger.info(f"Encontradas {len(resultado.get('anomalias', []))} anomalias")
        
        return jsonify({
            'sucesso': True,
            'resultado': resultado
        })
        
    except Exception as e:
        logger.error(f"Erro ao detectar anomalias: {str(e)}")
        return jsonify({
            'sucesso': False,
            'erro': str(e)
        }), 500


@app.route('/api/detectar-anomalias-historico', methods=['POST'])
def detectar_anomalias_historico():
    """
    Detecta anomalias considerando histórico do cliente.
    
    Exemplo de entrada:
    {
        "cargas": [...],  # Cargas atuais
        "historico": [...]  # Cargas anteriores (opcional)
    }
    """
    try:
        dados = request.get_json()
        
        if not dados or 'cargas' not in dados:
            return jsonify({
                'sucesso': False,
                'erro': 'Dados inválidos. Envie {"cargas": [...], "historico": [...]}'
            }), 400
        
        cargas = dados['cargas']
        historico = dados.get('historico', [])
        
        logger.info(f"Analisando {len(cargas)} cargas com histórico de {len(historico)} registros...")
        
        # Realizar análise com histórico
        resultado = detector_cliente.analisar_com_historico(cargas, historico)
        
        logger.info(f"Encontradas {len(resultado.get('anomalias', []))} anomalias")
        
        return jsonify({
            'sucesso': True,
            'resultado': resultado
        })
        
    except Exception as e:
        logger.error(f"Erro: {str(e)}")
        return jsonify({
            'sucesso': False,
            'erro': str(e)
        }), 500


# Executa a aplicação
if __name__ == '__main__':
    logger.info("Iniciando Shinsei Logistics API v2.0...")
    app.run(debug=True, host='0.0.0.0', port=5000)