#!/bin/bash
# Ejecutar en la Raspberry Pi 5 con Raspberry Pi OS 64-bit

set -e

echo "=== Instalando dependencias del sistema ==="
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip \
    libportaudio2 portaudio19-dev git

echo "=== Clonando proyecto ==="
cd ~
git clone <URL_DEL_REPO> sculpture-ai
cd sculpture-ai

echo "=== Creando entorno virtual ==="
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

echo "=== Configurando .env ==="
cp .env.example .env
echo "Edita ~/sculpture-ai/.env con tus API keys antes de continuar."
echo "Luego ejecuta: sudo systemctl enable sculpture && sudo systemctl start sculpture"
