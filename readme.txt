- Instrucciones para ejecutar el Pac-Man Safe RL -

PASO 1: Preparar el entorno

Abrid una terminal (Anaconda Prompt o PowerShell) en esta carpeta y ejecutad:

Crear un entorno limpio (opcional):
conda create -n pacman_rl python=3.10
conda activate pacman_rl

Instalar las librerías necesarias:
pip install -r requirements.txt

PASO 2: Descargar los juegos de Atari (¡IMPORTANTE!, me ha dado un dolor de cabeza tremendo)

Sin este paso, dará error porque no tenéis el juego de Ms. Pacman. Ejecutad esto en la terminal:

autorom --accept-license

(Esperad a que termine de descargar los juegos).

PASO 3: Ejecutar

python safe_pacman.py

Nota: El script primero entrenará (veréis una barra de progreso) y al terminar abrirá una ventana para ver jugar al Pac-Man to mono.