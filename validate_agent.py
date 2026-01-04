import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from safety_utils import PacmanSafetyMonitor 
import numpy as np
import pandas as pd
import time
import os

# Registrar entornos de Atari
gym.register_envs(ale_py)

# =========================================================
# ⚙️ CONFIGURACIÓN DEL TEST
# =========================================================
MODELO_A_CARGAR = "dqn_pacman_IA_Sola_ShieldON_Penalty_steps10000" 
CARPETA_MODELOS = "agentes_entrenados" 
NUM_EPISODIOS   = 5                
USAR_SHIELD     = True             
RENDERIZAR      = False            
CARPETA_SALIDA  = "validacion"     
# =========================================================

env_id = "ALE/MsPacman-v5"

# --- DEFINICIÓN DE WRAPPERS ---

class AddChannelDimWrapper(gym.ObservationWrapper):
    """Convierte (84, 84) -> (84, 84, 1). Necesario para DQN."""
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
    """
    Versión del Escudo específica para VALIDACIÓN.
    Añade 'safe_interventions' al diccionario 'info'.
    """
    def __init__(self, env):
        super().__init__(env)
        self.monitor = PacmanSafetyMonitor()
        self.episode_interventions = 0 
        
    def reset(self, **kwargs):
        self.episode_interventions = 0
        return self.env.reset(**kwargs)
        
    def step(self, action):
        pacman_pos, ghosts_pos = self.monitor.get_positions(self.env)
        is_unsafe, dist = self.monitor.is_danger(pacman_pos, ghosts_pos, threshold=25)
        
        final_action = action
        
        if is_unsafe:
            safe_action = self.monitor.get_safe_action(pacman_pos, ghosts_pos)
            final_action = safe_action
            self.episode_interventions += 1
            
        obs, reward, terminated, truncated, info = self.env.step(final_action)
        
        # Inyectamos el dato
        info['safe_interventions'] = self.episode_interventions
        
        return obs, reward, terminated, truncated, info

# ---------------------------------------------------------
# CONSTRUCCIÓN DEL ENTORNO
# ---------------------------------------------------------
def crear_entorno_validacion():
    modo = "human" if RENDERIZAR else None
    env = gym.make(env_id, frameskip=1, render_mode=modo)
    
    env = gym.wrappers.AtariPreprocessing(env, noop_max=0, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True)
    env = AddChannelDimWrapper(env)
    
    if USAR_SHIELD:
        env = SafeShieldWrapper(env)
    
    return env

# =========================================================
# EJECUCIÓN DEL TEST
# =========================================================
if __name__ == "__main__":
    val_env = DummyVecEnv([crear_entorno_validacion])
    val_env = VecFrameStack(val_env, n_stack=4)
    val_env = VecTransposeImage(val_env)

    # Construir ruta completa
    model_path = os.path.join(CARPETA_MODELOS, f"{MODELO_A_CARGAR}.zip")
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: No encuentro el archivo: {model_path}")
        exit()

    print(f"📂 Cargando modelo: {model_path} ...")
    model = DQN.load(model_path)
    print("✅ Modelo cargado.")

    if not os.path.exists(CARPETA_SALIDA):
        os.makedirs(CARPETA_SALIDA)

    print(f"\n🚀 Iniciando Validación de {NUM_EPISODIOS} episodios...")
    print(f"🛡️ Estado del Escudo: {'ACTIVADO' if USAR_SHIELD else 'DESACTIVADO'}")

    resultados = []

    try:
        for i in range(NUM_EPISODIOS):
            obs = val_env.reset()
            done = False
            total_reward = 0
            steps = 0
            intervenciones_finales = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = val_env.step(action)
                
                total_reward += reward
                steps += 1
                
                if USAR_SHIELD and 'safe_interventions' in info[0]:
                    intervenciones_finales = info[0]['safe_interventions']
                    
                if RENDERIZAR:
                    time.sleep(0.02) 

            # CÁLCULO DE MÉTRICAS NUEVAS
            eficiencia = total_reward[0] / steps if steps > 0 else 0
            ratio_seguridad = 100 * (1 - (intervenciones_finales / steps)) if steps > 0 else 0

            print(f"   🔹 Episodio {i+1}: Puntos={total_reward[0]:.0f} | Intervenciones={intervenciones_finales} | Eficiencia={eficiencia:.2f}")
            
            resultados.append({
                "Episodio": i+1,
                "Recompensa": float(total_reward[0]),
                "Duracion": steps,
                "Intervenciones": intervenciones_finales,
                "Eficiencia_Pto_Paso": eficiencia,    # NUEVA
                "Ratio_Seguridad_IA": ratio_seguridad, # NUEVA
                "Con_Escudo": USAR_SHIELD,
                "Modelo": MODELO_A_CARGAR
            })

    except KeyboardInterrupt:
        print("\n🛑 Validación detenida manualmente.")

    val_env.close()

    if resultados:
        df = pd.DataFrame(resultados)
        print("\n📊 --- RESUMEN DE VALIDACIÓN ---")
        print(f"Media de Puntos:        {df['Recompensa'].mean():.2f}")
        print(f"Media de Intervenciones: {df['Intervenciones'].mean():.2f}")
        print(f"Media de Eficiencia:     {df['Eficiencia_Pto_Paso'].mean():.3f} pts/paso")
        print(f"Seguridad Media (IA):    {df['Ratio_Seguridad_IA'].mean():.1f}%")
        
        nombre_archivo = f"validacion_{MODELO_A_CARGAR}_shield{USAR_SHIELD}.csv"
        ruta_completa = os.path.join(CARPETA_SALIDA, nombre_archivo)
        
        df.to_csv(ruta_completa, index=False)
        print(f"📝 Resultados guardados en: {ruta_completa}")
    else:
        print("No se completó ningún episodio.")