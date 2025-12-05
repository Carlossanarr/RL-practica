"""
-----------------------------------------------------------------------
🛡️ SAFE RL NIVEL 1: REWARD SHAPING (PENALIZACIÓN POR INTERVENCIÓN)
-----------------------------------------------------------------------
CONCEPTO:
En el código original, cuando el escudo salva a Pac-Man, el agente sigue
recibiendo recompensas positivas por sobrevivir. Esto crea un problema:
el agente no aprende que su acción original era suicida, porque el
escudo "lo tapa".

SOLUCIÓN:
Este script introduce una 'Negative Reward' (Castigo). Cada vez que el
Wrapper de seguridad se ve obligado a intervenir, modificamos la
recompensa a un valor negativo (ej. -5.0).

OBJETIVO:
Que el agente asocie "activar el escudo" con "dolor", forzándole a
aprender a jugar seguro por sí mismo para evitar la penalización.
-----------------------------------------------------------------------
"""

import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from safety_utils import PacmanSafetyMonitor 
import numpy as np

# Registrar entornos
gym.register_envs(ale_py)

# Reutilizamos tu wrapper de dimensiones del archivo original
class AddChannelDimWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if len(self.observation_space.shape) == 2:
            h, w = self.observation_space.shape
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8)
    def observation(self, obs):
        if len(obs.shape) == 2: return np.expand_dims(obs, axis=-1)
        return obs

# --- NUEVO WRAPPER NIVEL 1: PENALIZACIÓN ---
class PenaltyShieldWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.monitor = PacmanSafetyMonitor()
        
    def step(self, action):
        pacman_pos, ghosts_pos = self.monitor.get_positions(self.env)
        # Usamos el mismo umbral que tu script original
        is_unsafe, dist = self.monitor.is_danger(pacman_pos, ghosts_pos, threshold=25)
        
        if is_unsafe:
            # El escudo toma el control
            safe_action = self.monitor.get_safe_action(pacman_pos, ghosts_pos)
            
            # Ejecutamos la acción segura
            obs, reward, done, truncated, info = self.env.step(safe_action)
            
            # CAMBIO CLAVE: Castigamos al agente por necesitar ayuda.
            # Sobrescribimos la recompensa del juego con un valor negativo fuerte.
            # Así la IA asocia "activar escudo" con "dolor".
            penalty_reward = -5.0 
            return obs, penalty_reward, done, truncated, info
            
        return self.env.step(action)

# --- SETUP DE ENTRENAMIENTO (Igual que antes pero con el nuevo Wrapper) ---
def crear_entorno_nivel_1():
    env = gym.make("ALE/MsPacman-v5", frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True)
    env = AddChannelDimWrapper(env)
    env = PenaltyShieldWrapper(env) # Usamos el wrapper con castigo
    env = Monitor(env)
    return env

print("🛡️ NIVEL 1: Entrenando con Penalización por Seguridad...")
vec_env = DummyVecEnv([crear_entorno_nivel_1])
env = VecFrameStack(vec_env, n_stack=4)
env = VecTransposeImage(env)

model = DQN("CnnPolicy", env, verbose=1, buffer_size=10000, learning_starts=1000, exploration_fraction=0.2)
model.learn(total_timesteps=20000, progress_bar=True)
model.save("dqn_pacman_lvl1")
print("✅ Nivel 1 Completado.")

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