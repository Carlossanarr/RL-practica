import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from safe_pacman import AddChannelDimWrapper, SafeShieldWrapper 
import numpy as np
import pandas as pd
import time

# Registrar entornos
gym.register_envs(ale_py)

# =========================================================
# ⚙️ CONFIGURACIÓN DEL EXAMEN
# =========================================================
MODELO_A_CARGAR = "dqn_imitation_True_shield_False" # O "dqn_imitation_True_shield_True", etc.
NUM_EPISODIOS   = 10                # Sube a 50 o 100 para resultados serios
USAR_SHIELD     = False              # ¿Validamos CON o SIN la ayuda del escudo?
RENDERIZAR      = False             # False para ir rápido, True para ver
# =========================================================

env_id = "ALE/MsPacman-v5"

def crear_entorno_validacion():
    # render_mode=None para ir rápido, "human" para ver
    modo = "human" if RENDERIZAR else None
    env = gym.make(env_id, frameskip=1, render_mode=modo)
    
    # Preprocesamiento IGUAL que en entrenamiento
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True)
    env = AddChannelDimWrapper(env)
    
    # Opcional: ¿Activamos el escudo en el examen?
    if USAR_SHIELD:
        env = SafeShieldWrapper(env)
    
    return env

# Preparar entorno
val_env = DummyVecEnv([crear_entorno_validacion])
val_env = VecFrameStack(val_env, n_stack=4)
val_env = VecTransposeImage(val_env)

# Cargar Modelo
print(f"📂 Cargando modelo: {MODELO_A_CARGAR}...")
try:
    model = DQN.load(MODELO_A_CARGAR)
except:
    print("❌ No se encuentra el archivo del modelo. Asegúrate de haber entrenado primero.")
    exit()

print(f"🚀 Iniciando Validación de {NUM_EPISODIOS} episodios...")
print(f"🛡️ Estado del Escudo: {'ACTIVADO' if USAR_SHIELD else 'DESACTIVADO (A pelo)'}")

resultados = []

for i in range(NUM_EPISODIOS):
    obs = val_env.reset()
    done = False
    total_reward = 0
    steps = 0
    intervenciones = 0
    
    while not done:
        # PREDICCIÓN DETERMINISTA (Sin exploración aleatoria)
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, done, info = val_env.step(action)
        
        total_reward += reward
        steps += 1
        
        # Si usamos escudo, podemos leer cuántas veces nos salvó
        # info es una lista porque el entorno está vectorizado
        if USAR_SHIELD and 'safe_interventions' in info[0]:
            intervenciones = info[0]['safe_interventions']
            
        # En vectorizado, 'done' es True cuando acaba el episodio y se resetea solo.
        # Tenemos que romper el bucle manualmente.
        if done[0]:
            break

    print(f"Episodio {i+1}: Puntos={total_reward[0]}, Steps={steps}, Intervenciones={intervenciones}")
    
    resultados.append({
        "Episodio": i+1,
        "Recompensa": float(total_reward[0]),
        "Duracion": steps,
        "Intervenciones": intervenciones,
        "Con_Escudo": USAR_SHIELD
    })

val_env.close()

# Guardar y Mostrar Resumen
df = pd.DataFrame(resultados)
print("\n📊 --- RESUMEN DE VALIDACIÓN ---")
print(f"Media de Puntos:      {df['Recompensa'].mean():.2f} +/- {df['Recompensa'].std():.2f}")
print(f"Media de Duración:    {df['Duracion'].mean():.2f}")
print(f"Total Intervenciones: {df['Intervenciones'].sum()}")

nombre_csv = f"validacion_{MODELO_A_CARGAR}_shield_{USAR_SHIELD}.csv"
df.to_csv(nombre_csv, index=False)
print(f"📝 Resultados guardados en {nombre_csv}")