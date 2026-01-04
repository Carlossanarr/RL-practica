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
# ⚙️ CONFIGURACIÓN DEL EXAMEN
# =========================================================
# Asegúrate de poner el nombre EXACTO del archivo .zip que generó el entrenamiento (SIN la extensión .zip)
MODELO_A_CARGAR = "dqn_pacman_IA_Sola_ShieldON_Penalty_steps10000" 
CARPETA_MODELOS = "agentes_entrenados" # <--- Carpeta donde están los modelos entrenados
NUM_EPISODIOS   = 5                # Episodios para sacar la media
USAR_SHIELD     = True             # ¿Validamos CON o SIN la ayuda del escudo?
RENDERIZAR      = False            # True para ver jugar a la IA
CARPETA_SALIDA  = "validacion"     # Nombre de la carpeta para guardar resultados
# =========================================================

env_id = "ALE/MsPacman-v5"

# --- DEFINICIÓN DE WRAPPERS (Copiados y adaptados para Validación) ---

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
    Añade 'safe_interventions' al diccionario 'info' para poder contar las intervenciones.
    """
    def __init__(self, env):
        super().__init__(env)
        self.monitor = PacmanSafetyMonitor()
        self.episode_interventions = 0 # Contador por episodio
        
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
        
        # --- CLAVE: Inyectamos el dato para que el script de validación lo lea ---
        info['safe_interventions'] = self.episode_interventions
        
        return obs, reward, terminated, truncated, info

# ---------------------------------------------------------
# CONSTRUCCIÓN DEL ENTORNO
# ---------------------------------------------------------
def crear_entorno_validacion():
    # render_mode=None para ir rápido, "human" para ver
    modo = "human" if RENDERIZAR else None
    env = gym.make(env_id, frameskip=1, render_mode=modo)
    
    # Preprocesamiento IGUAL que en entrenamiento
    env = gym.wrappers.AtariPreprocessing(env, noop_max=0, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True)
    env = AddChannelDimWrapper(env)
    
    # Activamos el escudo solo si la configuración lo pide
    if USAR_SHIELD:
        env = SafeShieldWrapper(env)
    
    return env

# =========================================================
# EJECUCIÓN DEL TEST
# =========================================================
if __name__ == "__main__":
    # Preparar entorno vectorizado
    val_env = DummyVecEnv([crear_entorno_validacion])
    val_env = VecFrameStack(val_env, n_stack=4)
    val_env = VecTransposeImage(val_env)

    # Construir ruta completa del modelo
    model_path = os.path.join(CARPETA_MODELOS, f"{MODELO_A_CARGAR}.zip")
    
    # --- BLOQUE DE DEPURACIÓN DE RUTAS ---
    print("\n🔍 --- DIAGNÓSTICO DE RUTAS ---")
    print(f"📍 Directorio de trabajo actual (CWD): {os.getcwd()}")
    print(f"📂 Buscando archivo relativo: {model_path}")
    print(f"🗺️  Ruta absoluta calculada: {os.path.abspath(model_path)}")
    
    if not os.path.exists(model_path):
        print(f"\n❌ ERROR CRÍTICO: Python NO encuentra el archivo.")
        
        # Comprobar si al menos la carpeta existe
        if os.path.exists(CARPETA_MODELOS):
            print(f"✅ La carpeta '{CARPETA_MODELOS}' SÍ existe. Contenido:")
            archivos = os.listdir(CARPETA_MODELOS)
            if not archivos:
                print("   (La carpeta está vacía)")
            for f in archivos:
                print(f"   📄 {f}")
            print("\n💡 SUGERENCIA: Copia uno de los nombres de arriba (sin .zip) en 'MODELO_A_CARGAR'.")
        else:
            print(f"❌ La carpeta '{CARPETA_MODELOS}' NO existe en el directorio actual.")
            print("   Verifica que estás ejecutando el script desde la raíz del proyecto.")
        
        exit()
    # -------------------------------------

    model = DQN.load(model_path)
    print("✅ Modelo cargado correctamente.")

    # Crear carpeta de salida si no existe
    if not os.path.exists(CARPETA_SALIDA):
        os.makedirs(CARPETA_SALIDA)
        print(f"📁 Carpeta '{CARPETA_SALIDA}' creada (si no existía).")

    print(f"\n🚀 Iniciando Validación de {NUM_EPISODIOS} episodios...")
    print(f"🛡️ Estado del Escudo: {'ACTIVADO' if USAR_SHIELD else 'DESACTIVADO (A pelo)'}")

    resultados = []

    try:
        for i in range(NUM_EPISODIOS):
            obs = val_env.reset()
            done = False
            total_reward = 0
            steps = 0
            intervenciones_finales = 0
            
            while not done:
                # PREDICCIÓN DETERMINISTA (Sin exploración aleatoria, la IA juega en serio)
                action, _ = model.predict(obs, deterministic=True)
                
                obs, reward, done, info = val_env.step(action)
                
                total_reward += reward
                steps += 1
                
                # Leemos las intervenciones desde el info que modificamos en el wrapper
                if USAR_SHIELD and 'safe_interventions' in info[0]:
                    intervenciones_finales = info[0]['safe_interventions']
                    
                # Control de velocidad para que lo veas bien si está renderizando
                if RENDERIZAR:
                    time.sleep(0.02) 

            print(f"   🔹 Episodio {i+1}: Puntos={total_reward[0]:.0f} | Intervenciones={intervenciones_finales}")
            
            resultados.append({
                "Episodio": i+1,
                "Recompensa": float(total_reward[0]),
                "Duracion": steps,
                "Intervenciones": intervenciones_finales,
                "Con_Escudo": USAR_SHIELD,
                "Modelo": MODELO_A_CARGAR
            })

    except KeyboardInterrupt:
        print("\n🛑 Validación detenida manualmente.")

    val_env.close()

    # Guardar y Mostrar Resumen
    if resultados:
        df = pd.DataFrame(resultados)
        print("\n📊 --- RESUMEN DE VALIDACIÓN ---")
        print(f"Media de Puntos:      {df['Recompensa'].mean():.2f} +/- {df['Recompensa'].std():.2f}")
        print(f"Media de Intervenciones: {df['Intervenciones'].mean():.2f}")
        
        nombre_archivo = f"validacion_{MODELO_A_CARGAR}_shield{USAR_SHIELD}.csv"
        ruta_completa = os.path.join(CARPETA_SALIDA, nombre_archivo)
        
        df.to_csv(ruta_completa, index=False)
        print(f"📝 Resultados guardados en: {ruta_completa}")
    else:
        print("No se completó ningún episodio.")