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
PASOS_HUMANOS = 7500          
PASOS_ENTRENAMIENTO = 400000
LOG_INTERVALO = 10000 # Cada cuántos pasos guardamos una fila en el CSV
ENV_ID = "ALE/MsPacman-v5"
CARPETA_SALIDA = "agentes_entrenados" 
ARCHIVO_CSV_LOGS = "historial_training.csv" 

# --- CONFIGURACIÓN DE SEGURIDAD ---
USAR_ESCUDO_IA = False          
DISTANCIA_ESCUDO = 10           

# --- CONFIGURACIÓN DE RECOMPENSA ---
PENALIZAR_PELIGRO = True        
DISTANCIA_RECOMPENSA = 20       
PENALIZACION = -5.0            

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
        
        info['safe_interventions'] = self.interventions
        info['safe_penalties'] = self.penalties_applied
        
        return obs, reward, terminated, truncated, info

    def close(self):
        if not MODO_SOLO_HUMANO: 
            print("\n" + "="*45)
            print("📊 ESTADÍSTICAS FINALES (Consola)")
            if PENALIZAR_PELIGRO:
                print(f" 📉 Veces Penalizado: {self.penalties_applied}")
            if USAR_ESCUDO_IA:
                print(f" 🛡️ Intervenciones Escudo: {self.interventions}")
            print("="*45 + "\n")
        return super().close()

# ==========================================
# 2. CALLBACK DE MÉTRICAS (DATA COLLECTOR)
# ==========================================
class MetricsLoggingCallback(BaseCallback):
    def __init__(self, log_interval=5000, verbose=0):
        super(MetricsLoggingCallback, self).__init__(verbose)
        self.log_interval = log_interval
        self.total_deaths = 0
        self.total_reward_accumulated = 0.0 
        self.last_lives = None
        
        # Almacenaremos una lista de diccionarios (cada uno es una fila del CSV)
        self.rows_buffer = []

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        reward = self.locals['rewards'][0]
        
        self.total_reward_accumulated += reward
        
        current_lives = info.get('lives', 0)
        if self.last_lives is None: 
            self.last_lives = current_lives
            
        if current_lives < self.last_lives:
            self.total_deaths += (self.last_lives - current_lives)
        
        self.last_lives = current_lives
        
        # Registrar datos cada X pasos
        if self.num_timesteps % self.log_interval == 0:
            steps = self.num_timesteps
            current_interventions = info.get('safe_interventions', 0)
            current_penalties = info.get('safe_penalties', 0)
            
            efficiency = self.total_reward_accumulated / steps if steps > 0 else 0
            ppm = (self.total_reward_accumulated / self.total_deaths) if self.total_deaths > 0 else self.total_reward_accumulated
            
            # Guardamos los datos puros en el buffer
            self.rows_buffer.append({
                "Step": steps,
                "Deaths": self.total_deaths,
                "Reward_Accumulated": self.total_reward_accumulated,
                "Efficiency": efficiency,
                "PPM": ppm,
                "Interventions": current_interventions,
                "Penalties": current_penalties
            })
            
            if self.verbose > 0:
                print(f"Step {steps}: Deaths={self.total_deaths} | Eff={efficiency:.3f} | Int={current_interventions}")
                
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
                if dones[0]: obs = env.reset()
        except KeyboardInterrupt: pass
        print("\n✅ ¡Fase Humana Completada!")
        env.close()

    # --- FASE 2: ENTRENAMIENTO IA ---
    MODO_SOLO_HUMANO = False 
    print(f"\n🚀 Iniciando entrenamiento DQN ({PASOS_ENTRENAMIENTO} pasos)...")
    
    env = obtener_entorno_vectorizado(render_mode=None)
    if 'model' in locals():
        model.set_env(env)
        model.learning_starts = 0 
    else:
        model = DQN("CnnPolicy", env, buffer_size=50000, learning_starts=1000, exploration_fraction=0.2)

    # Callback
    metrics_callback = MetricsLoggingCallback(log_interval=LOG_INTERVALO, verbose=1)

    try:
        model.learn(total_timesteps=PASOS_ENTRENAMIENTO, progress_bar=True, callback=metrics_callback)
    except KeyboardInterrupt:
        print("\n⚠️ Entrenamiento detenido manualmente.")

    # Guardar modelo
    tipo = "Imitation" if USAR_IMITATION_WARMUP else "IA_Sola"
    s_tag = f"ShieldON_d{DISTANCIA_ESCUDO}" if USAR_ESCUDO_IA else "ShieldOFF"
    r_tag = f"_Penalty_d{DISTANCIA_RECOMPENSA}" if PENALIZAR_PELIGRO else ""
    model_name = f"dqn_pacman_{tipo}_{s_tag}{r_tag}_steps{PASOS_ENTRENAMIENTO}"
    
    if not os.path.exists(CARPETA_SALIDA): os.makedirs(CARPETA_SALIDA)
    ruta_modelo = os.path.join(CARPETA_SALIDA, model_name)
    model.save(ruta_modelo)
    print(f"💾 Modelo guardado: {ruta_modelo}")
    env.close()

    # ==========================================
    # 5. GUARDADO FORMATO LARGO (TIDY DATA)
    # ==========================================
    print(f"\n📝 Guardando historial en formato largo (Filas por Step)...")
    
    # 1. Recuperamos los datos del callback (Lista de diccionarios con Steps)
    raw_rows = metrics_callback.rows_buffer
    
    # 2. Preparamos la info estática del modelo (Configuración)
    config_info = {
        "Model_ID": model_name, # Identificador único para agrupar luego en gráficas
        "Imitation": USAR_IMITATION_WARMUP,
        "Shield_Active": USAR_ESCUDO_IA,
        "Shield_Dist": DISTANCIA_ESCUDO if USAR_ESCUDO_IA else 0,
        "Reward_Active": PENALIZAR_PELIGRO,
        "Reward_Dist": DISTANCIA_RECOMPENSA if PENALIZAR_PELIGRO else 0,
        "Total_Steps_Planned": PASOS_ENTRENAMIENTO
    }
    
    # 3. Fusionamos cada fila de steps con la info del modelo
    final_rows = []
    for row in raw_rows:
        # Unimos los dos diccionarios (Config + Métricas del paso)
        merged_row = {**config_info, **row}
        final_rows.append(merged_row)
        
    new_df = pd.DataFrame(final_rows)

    # 4. Guardamos/Añadimos al CSV maestro
    if os.path.exists(ARCHIVO_CSV_LOGS):
        # Si existe, lo cargamos y añadimos las nuevas filas al final
        existing_df = pd.read_csv(ARCHIVO_CSV_LOGS)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(ARCHIVO_CSV_LOGS, index=False)
        print(f"✅ Se han añadido {len(new_df)} filas al archivo '{ARCHIVO_CSV_LOGS}'")
    else:
        # Si no existe, lo creamos
        new_df.to_csv(ARCHIVO_CSV_LOGS, index=False)
        print(f"✅ Archivo '{ARCHIVO_CSV_LOGS}' creado con {len(new_df)} filas.")

    print("👋 ¡Entrenamiento finalizado!")