import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from safety_utils import PacmanSafetyMonitor 
import numpy as np
import time
import keyboard
import os

# ==========================================
# 0. CONFIGURACIÓN
# ==========================================
USAR_IMITATION_WARMUP = False # ¿Juegas tú primero? (Warmup)
PASOS_HUMANOS = 3000          
PASOS_ENTRENAMIENTO = 100000  
ENV_ID = "ALE/MsPacman-v5"
CARPETA_SALIDA = "agentes_entrenados" 

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
    """Convierte (84, 84) -> (84, 84, 1)"""
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
        # 1. Si está jugando el humano, no intervenimos nunca
        if MODO_SOLO_HUMANO:
            return self.env.step(action)

        # 2. Análisis de Seguridad
        # Calculamos usando el umbral mayor para asegurarnos de obtener la distancia real 
        # si está dentro del rango de "visión" más amplio (generalmente el de recompensa)
        max_dist = max(self.dist_shield, self.dist_reward)
        pacman_pos, ghosts_pos = self.monitor.get_positions(self.env)
        
        # is_danger devuelve la distancia al fantasma más cercano
        is_unsafe_general, dist = self.monitor.is_danger(pacman_pos, ghosts_pos, threshold=max_dist)
        
        final_action = action
        reward_adjustment = 0.0

        # 3. Lógica de Penalización (REWARD SHAPING)
        # Se activa si estamos más cerca que el umbral de prudencia (ej: 50px)
        if PENALIZAR_PELIGRO and dist < self.dist_reward:
            # Penalización lineal: más cerca = más castigo
            severity = 1.0 - (dist / self.dist_reward)
            reward_adjustment = self.penalty_val * severity
            self.penalties_applied += 1

        # 4. Lógica del Escudo (INTERVENCIÓN)
        # Se activa si estamos más cerca que el umbral de emergencia (ej: 10px) y el escudo está ON
        if USAR_ESCUDO_IA and dist < self.dist_shield:
            safe_action = self.monitor.get_safe_action(pacman_pos, ghosts_pos)
            final_action = safe_action
            self.interventions += 1
            
        # Ejecutamos la acción (original o corregida)
        obs, reward, terminated, truncated, info = self.env.step(final_action)
        
        # Aplicamos el ajuste a la recompensa
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
# 2. CONSTRUCCIÓN DEL ENTORNO
# ==========================================
def crear_entorno(render_mode=None):
    env = gym.make(ENV_ID, frameskip=1, render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=0, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True)
    env = AddChannelDimWrapper(env)
    
    # Pasamos las variables globales al wrapper
    env = SafeShieldWrapper(env, 
                            dist_shield=DISTANCIA_ESCUDO, 
                            dist_reward=DISTANCIA_RECOMPENSA, 
                            penalty_val=PENALIZACION)
    env = Monitor(env)
    return env

def obtener_entorno_vectorizado(render_mode=None):
    vec_env = DummyVecEnv([lambda: crear_entorno(render_mode)])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecTransposeImage(vec_env)
    return vec_env

# ==========================================
# 3. CONTROL HUMANO (MEJORADO)
# ==========================================
def obtener_accion_humana():
    if keyboard.is_pressed('up'): return 1
    elif keyboard.is_pressed('right'): return 2
    elif keyboard.is_pressed('left'): return 3
    elif keyboard.is_pressed('down'): return 4
    return 0 

# ==========================================
# 4. EJECUCIÓN DIRECTA
# ==========================================
if __name__ == "__main__":
    
    print("\n⚙️ Inicializando entorno...")
    
    # FASE 1: JUEGO HUMANO
    if USAR_IMITATION_WARMUP:
        MODO_SOLO_HUMANO = True 
        
        print("\n" + "="*50)
        print(f"🎮 MODO ENTRENAMIENTO HÍBRIDO ACTIVO")
        print(f"Objetivo: Jugar {PASOS_HUMANOS} pasos para enseñar a la IA.")
        print("Usa las FLECHAS del teclado.")
        print("="*50 + "\n")
        
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
                    print(f"💀 Pac-Man murió. Reiniciando... ({current_steps}/{PASOS_HUMANOS})")
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupción manual detectada. Pasando al entrenamiento...")

        print("\n✅ ¡Fase Humana Completada!")
        env.close()

    # FASE 2: ENTRENAMIENTO IA
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
        print("ℹ️ Creando modelo nuevo (sin experiencia humana previa)...")
        model = DQN("CnnPolicy", env, buffer_size=50000, learning_starts=1000, exploration_fraction=0.2)

    try:
        model.learn(total_timesteps=PASOS_ENTRENAMIENTO, progress_bar=True)
    except KeyboardInterrupt:
        print("\n⚠️ Entrenamiento detenido por el usuario. Guardando progreso...")

    # Generación de nombre con TODOS los parámetros
    tipo_entreno = "Imitation" if USAR_IMITATION_WARMUP else "IA_Sola"
    
    if USAR_ESCUDO_IA:
        seguridad_tag = f"ShieldON_d{DISTANCIA_ESCUDO}"
    else:
        seguridad_tag = "ShieldOFF"
        
    if PENALIZAR_PELIGRO:
        penalty_tag = f"_Penalty_d{DISTANCIA_RECOMPENSA}"
    else:
        penalty_tag = ""
    
    nombre_archivo = f"dqn_pacman_{tipo_entreno}_{seguridad_tag}{penalty_tag}_steps{PASOS_ENTRENAMIENTO}"
    
    # --- GESTIÓN DE CARPETAS ---
    if not os.path.exists(CARPETA_SALIDA):
        os.makedirs(CARPETA_SALIDA)
        print(f"📁 Carpeta '{CARPETA_SALIDA}' creada.")
        
    ruta_completa = os.path.join(CARPETA_SALIDA, nombre_archivo)
    
    print(f"💾 Guardando modelo en: {ruta_completa}")
    model.save(ruta_completa)
    
    env.close()
    print("👋 ¡Hasta la próxima!")
