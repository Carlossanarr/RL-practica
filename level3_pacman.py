"""
-----------------------------------------------------------------------
🛡️ SAFE RL NIVEL 3: CURRICULUM LEARNING (ESCUDO DECRECIENTE)
-----------------------------------------------------------------------
CONCEPTO:
Si el escudo siempre está activo, el agente puede volverse "vago" y
depender eternamente de él, sin aprender nunca las dinámicas de peligro
reales del entorno.

SOLUCIÓN:
Aplicamos un decaimiento lineal a la probabilidad de intervención.
Al paso 0, el escudo protege al 100%.
Al paso 10,000, protege al 50%.
Al paso 20,000, protege al 0%.

OBJETIVO:
Es como quitarle los ruedines a una bicicleta. Al principio se evita la
frustración de morir constantemente, pero poco a poco se obliga al agente
a responsabilizarse de su propia supervivencia.
-----------------------------------------------------------------------
"""

import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from safety_utils import PacmanSafetyMonitor 
import numpy as np
import random

gym.register_envs(ale_py)

class AddChannelDimWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if len(self.observation_space.shape) == 2:
            h, w = self.observation_space.shape
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8)
    def observation(self, obs):
        if len(obs.shape) == 2: return np.expand_dims(obs, axis=-1)
        return obs

# --- NUEVO WRAPPER NIVEL 3: ESCUDO QUE SE DESVANECE ---
class CurriculumShieldWrapper(gym.Wrapper):
    def __init__(self, env, total_steps=20000):
        super().__init__(env)
        self.monitor = PacmanSafetyMonitor()
        self.total_steps_limit = total_steps
        self.current_step_count = 0
        
    def step(self, action):
        self.current_step_count += 1
        
        # Calculamos la "Salud del Escudo"
        # Empieza en 1.0 (100%) y baja linealmente hasta 0.0
        shield_probability = 1.0 - (self.current_step_count / self.total_steps_limit)
        shield_probability = max(0.0, shield_probability) # Nunca menos de 0
        
        pacman_pos, ghosts_pos = self.monitor.get_positions(self.env)
        is_unsafe, _ = self.monitor.is_danger(pacman_pos, ghosts_pos, threshold=25)
        
        # Solo activamos el escudo si es peligroso Y tenemos suerte (según la probabilidad actual)
        if is_unsafe and random.random() < shield_probability:
            safe_action = self.monitor.get_safe_action(pacman_pos, ghosts_pos)
            return self.env.step(safe_action)
            
        # Si el escudo "falla" (o ya no existe), el agente se enfrenta a las consecuencias
        return self.env.step(action)

def crear_entorno_nivel_3():
    env = gym.make("ALE/MsPacman-v5", frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True)
    env = AddChannelDimWrapper(env)
    # Configuramos el escudo para que desaparezca a los 20k pasos
    env = CurriculumShieldWrapper(env, total_steps=20000) 
    env = Monitor(env)
    return env

print("🛡️ NIVEL 3: Entrenando con Curriculum (El escudo desaparecerá poco a poco)...")
vec_env = DummyVecEnv([crear_entorno_nivel_3])
env = VecFrameStack(vec_env, n_stack=4)
env = VecTransposeImage(env)

model = DQN("CnnPolicy", env, verbose=1, buffer_size=10000, learning_starts=1000, exploration_fraction=0.2)
model.learn(total_timesteps=20000, progress_bar=True)
model.save("dqn_pacman_lvl3")
print("✅ Nivel 3 Completado.")

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