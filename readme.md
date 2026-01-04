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




## Proyecto Pac-Man RL: DQN Seguro & Híbrido

Este proyecto implementa un agente de Deep Q-Network (DQN) para jugar a Ms. Pac-Man utilizando la librería Stable Baselines 3.

Se diferencia de una implementación estándar por incluir técnicas avanzadas de Seguridad (Safety RL) y Aprendizaje por Imitación (Imitation Learning) para acelerar y asegurar el entrenamiento.

🚀 Características Principales

🎮 Entrenamiento Híbrido (Human-in-the-Loop):

Permite al usuario jugar una fase inicial ("Warmup") para llenar la memoria de la IA con partidas de calidad.

La IA aprende de tus movimientos antes de empezar a explorar por su cuenta.

🛡️ Escudo de Seguridad (Safety Shield):

Un "Teacher" o supervisor monitoriza la distancia entre Pac-Man y los fantasmas.

Intervención: Si la IA va a cometer un error fatal, el escudo sobreescribe la acción para salvarla.

📉 Moldeado de Recompensa (Reward Shaping):

Penalización por Peligro: Se puede configurar para castigar a la IA (-10 puntos) cada vez que entra en una zona de riesgo, independientemente de si el escudo la salva o no. Esto fomenta que aprenda a tener "miedo" por sí misma.
