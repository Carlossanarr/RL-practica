# Proyecto Pac-Man RL: DQN Seguro & Híbrido

Este proyecto implementa un agente de Deep Q-Network (DQN) para jugar a Ms. Pac-Man utilizando la librería Stable Baselines 3.

Se diferencia de una implementación estándar por incluir técnicas avanzadas de Seguridad (Safety RL) y Aprendizaje por Imitación (Imitation Learning) para acelerar y asegurar el entrenamiento.

🚀 Características Principales

- 🎮 Entrenamiento Híbrido (Human-in-the-Loop):

    Permite al usuario jugar una fase inicial ("Warmup") para llenar la memoria de la IA con partidas de calidad.

    La IA aprende de tus movimientos antes de empezar a explorar por su cuenta.

- 🛡️ Escudo de Seguridad (Safety Shield):

    Un "Teacher" o supervisor monitoriza la distancia entre Pac-Man y los fantasmas.

    Intervención: Si la IA va a cometer un error fatal, el escudo sobreescribe la acción para salvarla.

- 📉 Moldeado de Recompensa (Reward Shaping):

    Penalización por Peligro: Se puede configurar para castigar a la IA (-10 puntos) cada vez que entra en una zona de riesgo, independientemente de si el escudo la salva o no. Esto fomenta que aprenda a tener "miedo" por sí misma.


- Instrucciones para ejecutar el Pac-Man Safe RL -

## Instalación 

1. Clonar el repositorio

2.  Preparar el entorno virtual (recomendado)

```bash 
    conda create -n pacman_rl python=3.10
    conda activate pacman_rl
```
Instalar las librerías necesarias:

```bash
pip install -r requirements.txt
```

3. Descargar los juegos de Atari 

``` bash 
autorom --accept-license
```
Este paso puede llevar un rato

## Configuración

El comportamiento del entrenamiento se controla modificando las variables al inicio del archivo train_pacman.py:

```python
# ==========================================
# 0. CONFIGURACIÓN
# ==========================================
USAR_IMITATION_WARMUP = True   # True: Juegas tú primero. False: La IA entrena sola desde el principio.
PASOS_HUMANOS = 1000           # Cuántos frames jugarás tú (si la opción anterior es True).
PASOS_ENTRENAMIENTO = 10000    # Cuántos pasos entrenará la IA autónomamente.

# --- CONFIGURACIÓN DE SEGURIDAD ---
USAR_ESCUDO_IA = True          # True: El escudo corrige a la IA si está en peligro. False: La IA puede morir libremente.
PENALIZAR_PELIGRO = True       # True: Resta 10 puntos si los fantasmas están cerca (enseña prudencia).

```

## Ejecución

Ejecutar el script train_pacman.py para entrenar la configuración de elegida:

``` bash
python train_pacman.py
```

Flujo de Ejecución:

- Fase Humana (Si está activa): * Se abrirá una ventana con el juego. Usa las FLECHAS DEL TECLADO para moverte.

**Nota**: Debes tener la terminal seleccionada/activa para que detecte las teclas.

Al terminar los pasos definidos, la ventana se cerrará.

- Fase de IA: La IA comenzará a entrenar a máxima velocidad (sin renderizado visual para ir rápido). Verás una barra de progreso en la terminal.

- Finalización:

Se mostrarán estadísticas de seguridad (cuántas veces intervino el escudo o se penalizó).

El modelo se guardará automáticamente con un nombre descriptivo, por ejemplo:
dqn_pacman_Imitation_ShieldON_Penalty_steps10000.zip

## Validación

Una vez entrando un agente, se puede utilizar el archivo validate_agent.py para calcular métricas del agente:

``` bash
python validate_agent.py
```







