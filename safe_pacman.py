import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from safety_utils import PacmanSafetyMonitor 
import numpy as np
import time

# Registrar entornos
gym.register_envs(ale_py)

# --- WRAPPER 1: CORRECCIÓN DE DIMENSIONES --- (esto ha dado muchos problemas, cuidado al tocarlo)
class AddChannelDimWrapper(gym.ObservationWrapper):
    """
    Convierte (84, 84) -> (84, 84, 1).
    Necesario para que la pila de frames y la transposición funcionen bien.
    """
    def __init__(self, env):
        super().__init__(env)
        if len(self.observation_space.shape) == 2:
            h, w = self.observation_space.shape
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8)
    
    def observation(self, obs):
        if len(obs.shape) == 2:
            return np.expand_dims(obs, axis=-1)
        return obs

# --- WRAPPER 2: EL ESCUDO (Teacher) ---
class SafeShieldWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.monitor = PacmanSafetyMonitor()
        self.interventions = 0 
        
    def step(self, action):
        pacman_pos, ghosts_pos = self.monitor.get_positions(self.env)
        is_unsafe, dist = self.monitor.is_danger(pacman_pos, ghosts_pos, threshold=25)
        
        final_action = action
        
        if is_unsafe:
            safe_action = self.monitor.get_safe_action(pacman_pos, ghosts_pos)
            final_action = safe_action
            self.interventions += 1
            # print(f"🛡️ Escudo activo (Dist: {dist})") # Descomentar para debug
            
        return self.env.step(final_action)
    
    def get_wrapper_attr(self, name):
        if name == "interventions":
            return self.interventions
        return super().get_wrapper_attr(name)

# ---------------------------------------------------------
# 1. ENTRENAMIENTO
# ---------------------------------------------------------

env_id = "ALE/MsPacman-v5"

print("🛡️ Configurando entorno de ENTRENAMIENTO...")

def crear_entorno_seguro():
    env = gym.make(env_id, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True)
    env = AddChannelDimWrapper(env)
    env = SafeShieldWrapper(env)
    env = Monitor(env)
    return env

# Pipeline de vectorización para entrenamiento #problemillas
vec_env = DummyVecEnv([crear_entorno_seguro])
env = VecFrameStack(vec_env, n_stack=4)     # (84, 84, 4)
env = VecTransposeImage(env)                # (4, 84, 84) -> PyTorch Friendly

print("🧠 Inicializando DQN (Student)...")
model = DQN("CnnPolicy", 
            env, 
            verbose=1,
            buffer_size=10000,
            learning_starts=1000, 
            exploration_fraction=0.2
           )

print("🚀 Entrenando con el Teacher vigilando...")
model.learn(total_timesteps=20000, progress_bar=True)
print("✅ Entrenamiento finalizado.")

model.save("dqn_pacman_safe")
env.close()

# ---------------------------------------------------------
# 2. VISUALIZACIÓN (Gracias gemini no se que haria sin ti)
# ---------------------------------------------------------
print("\n---------------------------------------")
print("🍿 ¡Preparando visualización! (Mira la nueva ventana)")
print("---------------------------------------")

def crear_entorno_visual():
    # render_mode="human" abre la ventana del juego
    env = gym.make(env_id, frameskip=1, render_mode="human")
    
    # Aplicamos EXACTAMENTE los mismos wrappers para que el modelo "vea" lo mismo
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True)
    env = AddChannelDimWrapper(env)
    env = SafeShieldWrapper(env) # ¡Incluimos el escudo para verlo actuar en vivo!
    return env

# Replicamos el pipeline de dimensiones
visual_env = DummyVecEnv([crear_entorno_visual])
visual_env = VecFrameStack(visual_env, n_stack=4)
visual_env = VecTransposeImage(visual_env)

obs = visual_env.reset()

try:
    print("🎮 Jugando... (Pulsa Ctrl+C en la terminal para salir)")
    while True:
        # Usamos el modelo entrenado para predecir
        action, _ = model.predict(obs, deterministic=True)
        
        # Paso del entorno
        obs, reward, done, info = visual_env.step(action)
        
        # Pausa para que sea visible al ojo humano
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nVisualización detenida por el usuario.")

visual_env.close()
print("👋 Fin.")