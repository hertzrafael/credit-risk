#!/bin/sh

MODEL_PATH="${MODEL_PATH:-/app/temp/credit_risk_model.pkl}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Modelo não encontrado em $MODEL_PATH"
    echo "Treinando modelo..."

    python model/main.py
else
    echo "Modelo encontrado em $MODEL_PATH"
fi

exec "$@"