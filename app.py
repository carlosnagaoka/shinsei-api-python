# API Flask para gerenciamento de entregas
# Autor: Shinsei API

from flask import Flask, request, jsonify

# Inicializa a aplicação Flask
app = Flask(__name__)


@app.route('/api/status', methods=['GET'])
def status():
    """
    Rota de verificação de status da API.
    Retorna uma mensagem confirmando que a API está funcionando.
    """
    return jsonify({
        'mensagem': 'API funcionando',
        'status': 'ok'
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


# Executa a aplicação em modo debug
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
