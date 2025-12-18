# Shinsei API Python

API simples em Flask para gerenciamento de entregas.

## Instalacao

```bash
pip install -r requirements.txt
```

## Executar

```bash
python app.py
```

A API estara disponivel em `http://localhost:5000`

## Rotas

### GET /api/status

Verifica se a API esta funcionando.

**Resposta:**
```json
{
  "mensagem": "API funcionando",
  "status": "ok"
}
```

### POST /api/relatorio

Recebe dados de entregas e retorna estatisticas.

**Requisicao:**
```json
{
  "entregas": [
    {"id": 1, "valor": 100.0, "status": "entregue"},
    {"id": 2, "valor": 50.0, "status": "pendente"},
    {"id": 3, "valor": 75.0, "status": "entregue"},
    {"id": 4, "valor": 30.0, "status": "cancelado"}
  ]
}
```

**Resposta:**
```json
{
  "mensagem": "Relatorio gerado com sucesso",
  "estatisticas": {
    "total_entregas": 4,
    "entregues": 2,
    "pendentes": 1,
    "canceladas": 1,
    "valor_total": 255.0,
    "valor_entregue": 175.0,
    "taxa_sucesso": 50.0
  }
}
```

## Testando com curl

```bash
# Verificar status
curl http://localhost:5000/api/status

# Enviar relatorio
curl -X POST http://localhost:5000/api/relatorio \
  -H "Content-Type: application/json" \
  -d '{"entregas": [{"id": 1, "valor": 100, "status": "entregue"}]}'
```
