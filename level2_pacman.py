"""
-----------------------------------------------------------------------
🛡️ SAFE RL NIVEL 2: LOOK-AHEAD SHIELDING (EVASIÓN INTELIGENTE)
-----------------------------------------------------------------------
CONCEPTO:
La lógica de seguridad original en 'safety_utils.py' es reactiva y simple:
si el fantasma está a la derecha, muévete a la izquierda. Esto a menudo
provoca que Pac-Man se quede atrapado en esquinas o callejones sin salida.

SOLUCIÓN:
Implementamos una estrategia 'Maximin'. Antes de corregir la acción,
el escudo simula las 4 direcciones posibles y calcula cuál de ellas
maximiza la distancia mínima a TODOS los fantasmas cercanos.

OBJETIVO:
Mejorar la calidad de la supervisión. El "Maestro" ahora es más listo
y no solo evita el choque inmediato, sino que busca la mejor ruta de escape.
-----------------------------------------------------------------------
"""

import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from safety_utils import PacmanSafetyMonitor 
import numpy as np

gym.register_envs(ale_py)

# Copiamos el wrapper de dimensiones necesario
class AddChannelDimWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if len(self.observation_space.shape) == 2:
            h, w = self.observation_space.shape
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8)
    def observation(self, obs):
        if len(obs.shape) == 2: return np.expand_dims(obs, axis=-1)
        return obs

# --- NUEVO LOGICA NIVEL 2: MEJOR EVASIÓN ---
class SmartSafetyMonitor(PacmanSafetyMonitor):
    def get_best_escape_action(self, pacman_pos, ghosts_pos):
        """
        Evalúa las 4 direcciones y elige la que más nos aleja del peligro.
        """
        px, py = pacman_pos
        possible_moves = {
            1: (px, py - 2), # UP (Y disminuye hacia arriba en pantallas)
            2: (px + 2, py), # RIGHT
            3: (px - 2, py), # LEFT
            4: (px, py + 2)  # DOWN
        }
        
        best_action = 0
        max_safety_score = -1
        
        # Probamos cada acción hipotética
        for action, (next_x, next_y) in possible_moves.items():
            # Calculamos la distancia al fantasma MÁS CERCANO desde la nueva posición
            # Queremos MAXIMIZAR la distancia mínima (Maximin)
            min_dist_to_ghost = 999
            for gx, gy in ghosts_pos:
                dist = abs(next_x - gx) + abs(next_y - gy)
                if dist < min_dist_to_ghost:
                    min_dist_to_ghost = dist
            
            if min_dist_to_ghost > max_safety_score:
                max_safety_score = min_dist_to_ghost
                best_action = action
                
        return best_action

class SmartShieldWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Usamos el monitor mejorado
        self.monitor = SmartSafetyMonitor() 

    def step(self, action):
        pacman_pos, ghosts_pos = self.monitor.get_positions(self.env)
        is_unsafe, _ = self.monitor.is_danger(pacman_pos, ghosts_pos, threshold=30) # Umbral un poco mayor
        
        if is_unsafe:
            # Usamos la nueva lógica inteligente
            safe_action = self.monitor.get_best_escape_action(pacman_pos, ghosts_pos)
            return self.env.step(safe_action)
            
        return self.env.step(action)

def crear_entorno_nivel_2():
    env = gym.make("ALE/MsPacman-v5", frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True)
    env = AddChannelDimWrapper(env)
    env = SmartShieldWrapper(env)
    env = Monitor(env)
    return env

print("🛡️ NIVEL 2: Entrenando con Escudo Inteligente (Look-Around)...")
vec_env = DummyVecEnv([crear_entorno_nivel_2])
env = VecFrameStack(vec_env, n_stack=4)
env = VecTransposeImage(env)

model = DQN("CnnPolicy", env, verbose=1, buffer_size=10000, learning_starts=1000, exploration_fraction=0.2)
model.learn(total_timesteps=20000, progress_bar=True)
model.save("dqn_pacman_lvl2")
print("✅ Nivel 2 Completado.")

# ---------------------------------------------------------
# ✂️ PEGAR ESTO AL FINAL DEL SCRIPT PARA VERLO JUGAR
# ---------------------------------------------------------
import time

print("\n---------------------------------------")
print("🍿 ¡Preparando visualización! (Mira la nueva ventana)")
print("---------------------------------------")

# 1. Creamos un entorno específico para visualizar (render_mode="human")
#    NOTA: Usamos los mismos wrappers básicos para que la IA 'vea' lo mismo.
visual_env = gym.make("ALE/MsPacman-v5", frameskip=1, render_mode="human")
visual_env = gym.wrappers.AtariPreprocessing(visual_env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True)
visual_env = AddChannelDimWrapper(visual_env)

# OJO: Aquí decidimos si activamos el escudo en la visualización o no.
# Si quieres ver al agente "a pelo" (sin ayudas), comenta la siguiente línea.
# Si quieres ver cómo interviene el escudo (recomendado para Nivel 1 y 2), déjala.
visual_env = SafeShieldWrapper(visual_env) if 'SafeShieldWrapper' in globals() else visual_env
# Nota: Para el Nivel 1 se llamaba PenaltyShieldWrapper, y Nivel 2 SmartShieldWrapper.
# Si da error, simplemente quita el wrapper de seguridad aquí para ver solo a la IA.

visual_env = DummyVecEnv([lambda: visual_env])
visual_env = VecFrameStack(visual_env, n_stack=4)
visual_env = VecTransposeImage(visual_env)

obs = visual_env.reset()

try:
    print("🎮 Jugando... (Pulsa Ctrl+C en la terminal para salir)")
    while True:
        # Usamos el modelo que acabamos de entrenar (variable 'model')
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, done, info = visual_env.step(action)
        time.sleep(0.05) # Velocidad normal

except KeyboardInterrupt:
    print("\nVisualización detenida por el usuario.")

visual_env.close()
print("👋 Fin del programa.")