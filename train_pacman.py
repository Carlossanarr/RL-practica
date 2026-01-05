import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback 
from safety_utils import PacmanSafetyMonitor 
import numpy as np
import pandas as pd  
import time
import keyboard
import os

# ==========================================
# 0. CONFIGURACIÓN
# ==========================================
USAR_IMITATION_WARMUP = False # ¿Juegas tú primero? (Warmup)
PASOS_HUMANOS = 3000          
PASOS_ENTRENAMIENTO = 10000 
LOG_INTERVALO = 1000 # DEJAR FIJO POR FAVOR
ENV_ID = "ALE/MsPacman-v5"
CARPETA_SALIDA = "agentes_entrenados" 
ARCHIVO_CSV_LOGS = "historial_training_completo.csv" # <--- Nombre nuevo para diferenciarlo

# --- CONFIGURACIÓN DE SEGURIDAD (SHIELDING) ---
USAR_ESCUDO_IA = True # Si True: El código interviene para salvar a Pacman
DISTANCIA_ESCUDO = 10 # Distancia (píxeles) a la que salta el escudo

# --- CONFIGURACIÓN DE RECOMPENSA (REWARD SHAPING) ---
PENALIZAR_PELIGRO = True # Si True: Resta puntos si hay fantasmas cerca
DISTANCIA_RECOMPENSA = 25 # Distancia (píxeles) a la que empieza a penalizar
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

        # Lógica de Penalización
        if PENALIZAR_PELIGRO and dist < self.dist_reward:
            severity = 1.0 - (dist / self.dist_reward)
            reward_adjustment = self.penalty_val * severity
            self.penalties_applied += 1

        # Lógica del Escudo
        if USAR_ESCUDO_IA and dist < self.dist_shield:
            safe_action = self.monitor.get_safe_action(pacman_pos, ghosts_pos)
            final_action = safe_action
            self.interventions += 1
            
        obs, reward, terminated, truncated, info = self.env.step(final_action)
        reward += reward_adjustment
        
        # --- NUEVO: Inyectamos estadísticas en 'info' para el Callback ---
        info['safe_interventions'] = self.interventions
        info['safe_penalties'] = self.penalties_applied
        
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
# 2. CALLBACK DE MÉTRICAS AVANZADAS
# ==========================================
class MetricsLoggingCallback(BaseCallback):
    """
    Registra Muertes, Eficiencia, PPM, Intervenciones y Penalizaciones cada X pasos.
    """
    def __init__(self, log_interval=5000, verbose=0):
        super(MetricsLoggingCallback, self).__init__(verbose)
        self.log_interval = log_interval
        
        # Acumuladores internos
        self.total_deaths = 0
        self.total_reward_accumulated = 0.0 # Recompensa total sumada paso a paso
        self.last_lives = None
        
        # Historial de datos (diccionarios paso -> valor)
        self.history = {
            "Deaths": {},
            "Efficiency": {},
            "PPM": {},
            "Interventions": {},
            "Penalties": {}
        }

    def _on_step(self) -> bool:
        # Acceso a infos y rewards del vector (asumimos 1 ambiente)
        info = self.locals['infos'][0]
        reward = self.locals['rewards'][0]
        
        # 1. Acumular recompensa bruta
        self.total_reward_accumulated += reward
        
        # 2. Detectar Muertes
        current_lives = info.get('lives', 0)
        if self.last_lives is None: 
            self.last_lives = current_lives
            
        if current_lives < self.last_lives:
            diff = self.last_lives - current_lives
            self.total_deaths += diff
        
        self.last_lives = current_lives # Actualizamos vidas (si suben por reset, no pasa nada)
        
        # 3. Leer estadísticas del Wrapper
        current_interventions = info.get('safe_interventions', 0)
        current_penalties = info.get('safe_penalties', 0)
        
        # 4. Registrar Logs en Intervalos
        if self.num_timesteps % self.log_interval == 0:
            steps = self.num_timesteps
            
            # Cálculos
            efficiency = self.total_reward_accumulated / steps if steps > 0 else 0
            ppm = (self.total_reward_accumulated / self.total_deaths) if self.total_deaths > 0 else self.total_reward_accumulated
            
            # Guardar en historial
            self.history["Deaths"][steps] = self.total_deaths
            self.history["Efficiency"][steps] = efficiency
            self.history["PPM"][steps] = ppm
            self.history["Interventions"][steps] = current_interventions
            self.history["Penalties"][steps] = current_penalties
            
            if self.verbose > 0:
                print(f"Step {steps}: Deaths={self.total_deaths} | Eff={efficiency:.3f} | PPM={ppm:.1f} | Int={current_interventions}")
                
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

    # Inicializamos el Callback de Métricas Completas
    metrics_callback = MetricsLoggingCallback(log_interval=LOG_INTERVALO, verbose=1)

    try:
        model.learn(total_timesteps=PASOS_ENTRENAMIENTO, progress_bar=True, callback=metrics_callback)
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
    # 5. GUARDADO AVANZADO DE LOGS EN CSV
    # ==========================================
    print(f"\n📝 Procesando logs de entrenamiento...")
    
    # Datos básicos del modelo
    row_data = {
        "Model_Name": model_name,
        "Imitation": USAR_IMITATION_WARMUP,
        "Shield": USAR_ESCUDO_IA,
        "Shield_Dist": DISTANCIA_ESCUDO if USAR_ESCUDO_IA else 0,
        "Reward_Shaping": PENALIZAR_PELIGRO,
        "Reward_Dist": DISTANCIA_RECOMPENSA if PENALIZAR_PELIGRO else 0,
        "Total_Steps": PASOS_ENTRENAMIENTO
    }
    
    # Extraemos el historial del callback
    # Estructura: self.history["Deaths"][step] = valor
    metrics_data = metrics_callback.history
    
    # Obtenemos la lista de pasos registrados (e.g., 5000, 10000...)
    # Usamos "Deaths" como referencia, pero todos tienen las mismas claves
    registered_steps = sorted(metrics_data["Deaths"].keys())
    
    # Aplanamos los diccionarios en columnas: Step_5000_Deaths, Step_5000_Efficiency, etc.
    for step in registered_steps:
        for metric_name, values_dict in metrics_data.items():
            # Ejemplo de columna: Step_5000_Deaths
            col_name = f"Step_{step}_{metric_name}"
            row_data[col_name] = values_dict[step]
        
    new_df = pd.DataFrame([row_data])

    if os.path.exists(ARCHIVO_CSV_LOGS):
        # Leemos y concatenamos para añadir la fila
        existing_df = pd.read_csv(ARCHIVO_CSV_LOGS)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(ARCHIVO_CSV_LOGS, index=False)
        print(f"✅ Datos añadidos a '{ARCHIVO_CSV_LOGS}'")
    else:
        new_df.to_csv(ARCHIVO_CSV_LOGS, index=False)
        print(f"✅ Archivo '{ARCHIVO_CSV_LOGS}' creado con métricas detalladas.")

    print("👋 ¡Hasta la próxima!")