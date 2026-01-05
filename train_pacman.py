import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback # <--- Importante para el callback
from safety_utils import PacmanSafetyMonitor 
import numpy as np
import pandas as pd # <--- Necesario para guardar el CSV
import time
import keyboard
import os

# ==========================================
# 0. CONFIGURACIÓN
# ==========================================
USAR_IMITATION_WARMUP = False # ¿Juegas tú primero? (Warmup)
PASOS_HUMANOS = 3000          
PASOS_ENTRENAMIENTO = 25000 
LOG_INTERVALO = 5000 # <--- Cada cuántos pasos guardamos el dato de muertes (5k para mejor gráfica)
ENV_ID = "ALE/MsPacman-v5"
CARPETA_SALIDA = "agentes_entrenados" 
ARCHIVO_CSV_LOGS = "historial_muertes_training.csv" # <--- Nombre del archivo acumulativo

# --- CONFIGURACIÓN DE SEGURIDAD (SHIELDING) ---
USAR_ESCUDO_IA = True # Si True: El código interviene para salvar a Pacman
DISTANCIA_ESCUDO = 10 # Distancia (píxeles) a la que salta el escudo (Emergencia)

# --- CONFIGURACIÓN DE RECOMPENSA (REWARD SHAPING) ---
PENALIZAR_PELIGRO = True # Si True: Resta puntos si hay fantasmas cerca
DISTANCIA_RECOMPENSA = 25 # Distancia (píxeles) a la que empieza a penalizar (Prudencia)
PENALIZACION = -10.0 # Cuántos puntos restar

# VARIABLE DE ESTADO (No tocar)
MODO_SOLO_HUMANO = False 

gym.register_envs(ale_py)

# ==========================================
# 1. WRAPPERS
# ==========================================
class AddChannelDimWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if len(self.observation_space.shape) == 2:
            h, w = self.observation_space.shape
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8)
    
    def observation(self, obs):
        if len(obs.shape) == 2:
            return np.expand_dims(obs, axis=-1)
        return obs

class SafeShieldWrapper(gym.Wrapper):
    def __init__(self, env, dist_shield, dist_reward, penalty_val):
        super().__init__(env)
        self.monitor = PacmanSafetyMonitor()
        self.dist_shield = dist_shield
        self.dist_reward = dist_reward
        self.penalty_val = penalty_val
        self.interventions = 0   
        self.penalties_applied = 0 
        
    def step(self, action):
        if MODO_SOLO_HUMANO:
            return self.env.step(action)

        max_dist = max(self.dist_shield, self.dist_reward)
        pacman_pos, ghosts_pos = self.monitor.get_positions(self.env)
        is_unsafe_general, dist = self.monitor.is_danger(pacman_pos, ghosts_pos, threshold=max_dist)
        
        final_action = action
        reward_adjustment = 0.0

        if PENALIZAR_PELIGRO and dist < self.dist_reward:
            severity = 1.0 - (dist / self.dist_reward)
            reward_adjustment = self.penalty_val * severity
            self.penalties_applied += 1

        if USAR_ESCUDO_IA and dist < self.dist_shield:
            safe_action = self.monitor.get_safe_action(pacman_pos, ghosts_pos)
            final_action = safe_action
            self.interventions += 1
            
        obs, reward, terminated, truncated, info = self.env.step(final_action)
        reward += reward_adjustment
        return obs, reward, terminated, truncated, info

    def close(self):
        if not MODO_SOLO_HUMANO: 
            print("\n" + "="*45)
            print("📊 ESTADÍSTICAS DE SEGURIDAD (Post-Entreno)")
            if PENALIZAR_PELIGRO:
                print(f" 📉 Veces Penalizado (Dist < {self.dist_reward}): {self.penalties_applied}")
            else:
                print(f" 📉 Penalización desactivada (0)")
            if USAR_ESCUDO_IA:
                print(f" 🛡️ Intervenciones del Escudo (Dist < {self.dist_shield}): {self.interventions}")
            else:
                print(f" 🛡️ Escudo desactivado (0)")
            print("="*45 + "\n")
        return super().close()

# ==========================================
# 2. CALLBACK PARA CONTAR MUERTES
# ==========================================
class DeathLoggingCallback(BaseCallback):
    """
    Callback personalizado para registrar muertes acumuladas durante el entrenamiento.
    """
    def __init__(self, log_interval=5000, verbose=0):
        super(DeathLoggingCallback, self).__init__(verbose)
        self.log_interval = log_interval
        self.total_deaths = 0
        self.last_lives = None
        # Diccionario para guardar el historial: {step: muertes_acumuladas}
        self.death_history = {} 

    def _on_step(self) -> bool:
        # Accedemos a la info del entorno (vectorizado)
        infos = self.locals['infos']
        
        # Iteramos por los entornos (normalmente 1 en DummyVecEnv)
        current_lives = 0
        for info in infos:
            # ALE/Gymnasium suele devolver 'lives' en el diccionario info
            if 'lives' in info:
                current_lives += info['lives']
        
        # Inicialización
        if self.last_lives is None:
            self.last_lives = current_lives
            
        # Detectar muerte: Si las vidas bajan, sumamos al contador
        if current_lives < self.last_lives:
            diff = self.last_lives - current_lives
            self.total_deaths += diff
            
        # Si las vidas suben (reset del juego), actualizamos sin contar muerte
        self.last_lives = current_lives
        
        # Registrar datos según el intervalo
        if self.num_timesteps % self.log_interval == 0:
            self.death_history[self.num_timesteps] = self.total_deaths
            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: Muertes acumuladas = {self.total_deaths}")
                
        return True

# ==========================================
# 3. FUNCIONES DE ENTORNO
# ==========================================
def crear_entorno(render_mode=None):
    env = gym.make(ENV_ID, frameskip=1, render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=0, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True)
    env = AddChannelDimWrapper(env)
    env = SafeShieldWrapper(env, dist_shield=DISTANCIA_ESCUDO, dist_reward=DISTANCIA_RECOMPENSA, penalty_val=PENALIZACION)
    env = Monitor(env)
    return env

def obtener_entorno_vectorizado(render_mode=None):
    vec_env = DummyVecEnv([lambda: crear_entorno(render_mode)])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecTransposeImage(vec_env)
    return vec_env

def obtener_accion_humana():
    if keyboard.is_pressed('up'): return 1
    elif keyboard.is_pressed('right'): return 2
    elif keyboard.is_pressed('left'): return 3
    elif keyboard.is_pressed('down'): return 4
    return 0 

# ==========================================
# 4. EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    
    print("\n⚙️ Inicializando entorno...")
    
    # --- FASE 1: HUMAN WARMUP ---
    if USAR_IMITATION_WARMUP:
        MODO_SOLO_HUMANO = True 
        print(f"\n🎮 MODO ENTRENAMIENTO HÍBRIDO ACTIVO ({PASOS_HUMANOS} pasos)")
        
        env = obtener_entorno_vectorizado(render_mode="human")
        model = DQN("CnnPolicy", env, buffer_size=50000, learning_starts=1000, exploration_fraction=0.2)
        obs = env.reset()
        current_steps = 0
        try:
            while current_steps < PASOS_HUMANOS:
                action_int = obtener_accion_humana()
                action_array = np.array([action_int]) 
                next_obs, rewards, dones, infos = env.step(action_array)
                model.replay_buffer.add(obs, next_obs, action_array, rewards, dones, infos)
                obs = next_obs
                current_steps += 1
                time.sleep(0.04) 
                if dones[0]:
                    obs = env.reset()
        except KeyboardInterrupt:
            print("\n⚠️ Interrupción manual en Warmup.")
        print("\n✅ ¡Fase Humana Completada!")
        env.close()

    # --- FASE 2: ENTRENAMIENTO IA ---
    MODO_SOLO_HUMANO = False 
    
    print(f"\n🛡️ ESTADO DE SEGURIDAD:")
    print(f" - Escudo: {'ON' if USAR_ESCUDO_IA else 'OFF'} (Activa a {DISTANCIA_ESCUDO} px)")
    print(f" - Recompensa: {'ON' if PENALIZAR_PELIGRO else 'OFF'} (Penaliza a {DISTANCIA_RECOMPENSA} px)")
    print("🚀 Iniciando entrenamiento DQN...")
    
    env = obtener_entorno_vectorizado(render_mode=None)
    
    if 'model' in locals():
        model.set_env(env)
        model.learning_starts = 0 
    else:
        print("ℹ️ Creando modelo nuevo...")
        model = DQN("CnnPolicy", env, buffer_size=50000, learning_starts=1000, exploration_fraction=0.2)

    # Inicializamos el Callback de Muertes
    death_callback = DeathLoggingCallback(log_interval=LOG_INTERVALO, verbose=1)

    try:
        # Pasamos el callback al método learn
        model.learn(total_timesteps=PASOS_ENTRENAMIENTO, progress_bar=True, callback=death_callback)
    except KeyboardInterrupt:
        print("\n⚠️ Entrenamiento detenido por el usuario.")

    # Generación de nombre
    tipo_entreno = "Imitation" if USAR_IMITATION_WARMUP else "IA_Sola"
    if USAR_ESCUDO_IA:
        seguridad_tag = f"ShieldON_d{DISTANCIA_ESCUDO}"
    else:
        seguridad_tag = "ShieldOFF"
        
    if PENALIZAR_PELIGRO:
        penalty_tag = f"_Penalty_d{DISTANCIA_RECOMPENSA}"
    else:
        penalty_tag = ""
    
    model_name = f"dqn_pacman_{tipo_entreno}_{seguridad_tag}{penalty_tag}_steps{PASOS_ENTRENAMIENTO}"
    
    # Guardar modelo
    if not os.path.exists(CARPETA_SALIDA):
        os.makedirs(CARPETA_SALIDA)
    ruta_modelo = os.path.join(CARPETA_SALIDA, model_name)
    print(f"💾 Guardando modelo en: {ruta_modelo}")
    model.save(ruta_modelo)
    
    env.close()

    # ==========================================
    # 5. GUARDADO DE LOGS EN CSV
    # ==========================================
    print(f"\n📝 Procesando logs de entrenamiento...")
    
    # Convertimos el historial del callback a un DataFrame
    # Estructura: Pasos (columnas) -> Muertes (valores)
    data = death_callback.death_history
    
    # Creamos una fila con el nombre del modelo y la configuración
    row_data = {
        "Model_Name": model_name,
        "Imitation": USAR_IMITATION_WARMUP,
        "Shield": USAR_ESCUDO_IA,
        "Shield_Dist": DISTANCIA_ESCUDO if USAR_ESCUDO_IA else 0,
        "Reward_Shaping": PENALIZAR_PELIGRO,
        "Reward_Dist": DISTANCIA_RECOMPENSA if PENALIZAR_PELIGRO else 0,
        "Total_Steps": PASOS_ENTRENAMIENTO
    }
    
    # Añadimos las columnas de pasos dinámicamente (step_5000, step_10000...)
    for step, deaths in data.items():
        row_data[f"Step_{step}"] = deaths
        
    new_df = pd.DataFrame([row_data])

    # Lógica de Append: Si existe, se añade; si no, se crea
    if os.path.exists(ARCHIVO_CSV_LOGS):
        # Leemos el existente para alinear columnas si hiciera falta
        existing_df = pd.read_csv(ARCHIVO_CSV_LOGS)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(ARCHIVO_CSV_LOGS, index=False)
        print(f"✅ Fila añadida a '{ARCHIVO_CSV_LOGS}'")
    else:
        new_df.to_csv(ARCHIVO_CSV_LOGS, index=False)
        print(f"✅ Archivo '{ARCHIVO_CSV_LOGS}' creado exitosamente.")

    print("👋 ¡Hasta la próxima!")