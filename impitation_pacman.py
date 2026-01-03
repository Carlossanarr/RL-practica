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
USAR_IMITATION_WARMUP = False  # ¿Juegas tú primero?
PASOS_HUMANOS = 1000          
PASOS_ENTRENAMIENTO = 10000  
ENV_ID = "ALE/MsPacman-v5"

# --- CONFIGURACIÓN DE SEGURIDAD ---
USAR_ESCUDO_IA = False          # Si True: El código sobreescribe la acción para salvar a Pacman
PENALIZAR_PELIGRO = True       # Si True: Resta puntos (-10) si hay fantasmas cerca (Independiente del escudo)

# VARIABLE DE ESTADO (NO TOCAAR)

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
    def __init__(self, env):
        super().__init__(env)
        self.monitor = PacmanSafetyMonitor()
        self.interventions = 0   # Veces que el escudo corrigió la acción
        self.danger_counts = 0   # Veces que se detectó peligro (para penalización)
        
    def step(self, action):
        # 1. Si está jugando el humano, no intervenimos nunca
        if MODO_SOLO_HUMANO:
            return self.env.step(action)

        # 2. Análisis de Seguridad (Siempre se calcula)
        pacman_pos, ghosts_pos = self.monitor.get_positions(self.env)
        is_unsafe, dist = self.monitor.is_danger(pacman_pos, ghosts_pos, threshold=25)
        
        final_action = action
        penalty = 0.0

        # 3. Lógica de Penalización (Independiente del Escudo)
        # Si la IA se mete en la boca del lobo, la castigamos aunque no tengamos escudo
        if is_unsafe:
            self.danger_counts += 1
            if PENALIZAR_PELIGRO:
                penalty = -10.0 

        # 4. Lógica del Escudo (Intervención)
        # Solo interviene si hay peligro Y el escudo está activado
        if is_unsafe and USAR_ESCUDO_IA:
            safe_action = self.monitor.get_safe_action(pacman_pos, ghosts_pos)
            final_action = safe_action
            self.interventions += 1
            
        # Ejecutamos la acción (original o corregida)
        obs, reward, terminated, truncated, info = self.env.step(final_action)
        
        # Aplicamos la penalización al reward
        reward += penalty
            
        return obs, reward, terminated, truncated, info

    def close(self):
        # Imprimir estadísticas al cerrar el entorno
        if not MODO_SOLO_HUMANO: # Solo mostrar stats de la sesión de IA
            print("\n" + "="*45)
            print("📊 ESTADÍSTICAS DE SEGURIDAD (Post-Entreno)")
            print(f"   ⚠️  Veces en Zona de Peligro: {self.danger_counts}")
            print(f"   🛡️  Intervenciones del Escudo: {self.interventions}")
            print("="*45 + "\n")
        return super().close()

# ==========================================
# 2. CONSTRUCCIÓN DEL ENTORNO
# ==========================================
def crear_entorno(render_mode=None):
    env = gym.make(ENV_ID, frameskip=1, render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=0, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True)
    env = AddChannelDimWrapper(env)
    env = SafeShieldWrapper(env)
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
    
    print("\n⚙️  Inicializando entorno...")
    
    # FASE 1: JUEGO HUMANO (Solo si USAR_IMITATION_WARMUP es True)
    if USAR_IMITATION_WARMUP:
        MODO_SOLO_HUMANO = True 
        
        print("\n" + "="*50)
        print(f"🎮 MODO ENTRENAMIENTO HÍBRIDO ACTIVO")
        print(f"Objetivo: Jugar {PASOS_HUMANOS} pasos para enseñar a la IA.")
        print("Usa las FLECHAS del teclado.")
        print("="*50 + "\n")
        
        # Creamos entorno visual
        env = obtener_entorno_vectorizado(render_mode="human")
        
        # Inicializamos modelo
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
    MODO_SOLO_HUMANO = False # Aseguramos que el escudo protege a la IA (si está activo)
    
    print(f"\n🛡️ ESTADO DE SEGURIDAD:")
    print(f"   - Escudo Activo (Intervención): {USAR_ESCUDO_IA}")
    print(f"   - Penalización por Riesgo (Reward Shaping): {PENALIZAR_PELIGRO}")
    print("🚀 Iniciando entrenamiento DQN...")
    
    # Recreamos el entorno sin ventana para ir rápido
    env = obtener_entorno_vectorizado(render_mode=None)
    
    # Si ya teníamos modelo (del warmup), solo cambiamos el entorno
    if 'model' in locals():
        model.set_env(env)
        model.learning_starts = 0 # Aprendemos YA de lo que jugaste
    else:
        # Si NO hubo warmup, creamos el modelo de cero aquí
        print("ℹ️ Creando modelo nuevo (sin experiencia humana previa)...")
        model = DQN("CnnPolicy", env, buffer_size=50000, learning_starts=1000, exploration_fraction=0.2)

    try:
        model.learn(total_timesteps=PASOS_ENTRENAMIENTO, progress_bar=True)
    except KeyboardInterrupt:
        print("\n⚠️ Entrenamiento detenido por el usuario. Guardando progreso...")

    # Generamos el nombre dinámico del archivo para que sepas qué configuración usaste
    tipo_entreno = "Hibrido" if USAR_IMITATION_WARMUP else "IA_Sola"
    seguridad_tag = "ShieldON" if USAR_ESCUDO_IA else "ShieldOFF"
    penalty_tag = "_Penalty" if PENALIZAR_PELIGRO else ""
    
    nombre_archivo = f"dqn_pacman_{tipo_entreno}_{seguridad_tag}{penalty_tag}_steps{PASOS_ENTRENAMIENTO}"
    
    print(f"💾 Guardando modelo como: {nombre_archivo}")
    model.save(nombre_archivo)
    
    # Al cerrar el env se imprimirán las estadísticas gracias al método close() del wrapper
    env.close()
    print("👋 ¡Hasta la próxima!")