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
MODELO_A_CARGAR = "dqn_pacman_IA_Sola_steps10000" 
CARPETA_MODELOS = "agentes_entrenados" 
NUM_EPISODIOS = 5                
RENDERIZAR = False # True para ver jugar a la IA
CARPETA_SALIDA = "validacion"     

# --- CONFIGURACIÓN DE SEGURIDAD (Debe coincidir con tu experimento) ---
USAR_SHIELD = True # ¿Activamos el escudo en la validación?
DISTANCIA_ESCUDO = 1 # IMPORTANTE: Poner  la misma distancia que usaste al entrenar (10, 25 o 50)

DISTANCIA_RECOMPENSA = 5 # Debe coincidir con el valor usado en entrenamiento

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
    Recibe el umbral de distancia dinámicamente.
    """
    def __init__(self, env, umbral_distancia):
        super().__init__(env)
        self.monitor = PacmanSafetyMonitor()
        self.umbral = umbral_distancia # <-- CORRECCIÓN: Usamos la variable, no un número fijo
        self.episode_interventions = 0 
        
    def reset(self, **kwargs):
        self.episode_interventions = 0
        return self.env.reset(**kwargs)
        
    def step(self, action):
        pacman_pos, ghosts_pos = self.monitor.get_positions(self.env)
        
        # Usamos el umbral configurado
        is_unsafe, dist = self.monitor.is_danger(pacman_pos, ghosts_pos, threshold=self.umbral)
        
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
    
    # Límite de tiempo extendido para que pueda ganar si es muy bueno
    env = gym.wrappers.TimeLimit(env, max_episode_steps=25000)

    env = AddChannelDimWrapper(env)
    
    if USAR_SHIELD:
        # CORRECCIÓN: Pasamos la distancia configurada
        env = SafeShieldWrapper(env, umbral_distancia=DISTANCIA_ESCUDO)
    
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
    if USAR_SHIELD:
        print(f"🛡️ Escudo: ACTIVADO (Distancia: {DISTANCIA_ESCUDO} px)")
    else:
        print(f"🛡️ Escudo: DESACTIVADO")

    resultados = []

    monitor_metricas = PacmanSafetyMonitor()

    try:

        for i in range(NUM_EPISODIOS):
            obs = val_env.reset()
            done = False
            total_reward = 0
            steps = 0
            intervenciones_finales = 0
            
            # --- Variables para métricas nuevas ---
            muertes = 0
            vidas_previas = None
            win = 0
            end_reason = "Unknown"
            unsafe_steps = 0

            # --------------------------------------

            while not done:

                # ------------ METRICA UNSAFE STEPS --------------
                # ---- Conteo de pasos inseguros (umbral DISTANCIA_RECOMPENSA) ----
                base_env0 = val_env.venv.envs[0]
                pacman_pos, ghosts_pos = monitor_metricas.get_positions(base_env0)


                #if steps == 0:
                #    print("DEBUG positions:", pacman_pos, ghosts_pos[:2], "n_ghosts=", len(ghosts_pos))


                is_unsafe_r, _ = monitor_metricas.is_danger(pacman_pos, ghosts_pos, threshold=DISTANCIA_RECOMPENSA)
                
                if is_unsafe_r:
                    unsafe_steps += 1

                # ------------------------------------
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = val_env.step(action)

                # --- LÓGICA DE MÉTRICAS NUEVAS ---


                info0 = info[0] # Accedemos al primer entorno del vector

                # Detección de Victoria vs Timeout
                time_limit_trunc = False
                if 'TimeLimit.truncated' in info0:
                    time_limit_trunc = bool(info0['TimeLimit.truncated'])
                elif 'truncated' in info0:
                    time_limit_trunc = bool(info0['truncated'])

                if done:
                    # Si se acaba el juego pero NO es por tiempo, asumimos Game Over
                    # Si se acaba POR tiempo (truncated), asumimos que sobrevivió (Win técnico)
                    win = 1 if time_limit_trunc else 0
                    end_reason = "TimeLimit" if time_limit_trunc else "GameOver"

                # Conteo de vidas perdidas
                if 'lives' in info0:
                    vidas_actuales = info0['lives']
                    if vidas_previas is None:
                        vidas_previas = vidas_actuales
                    else:
                        if vidas_actuales < vidas_previas:
                            muertes += (vidas_previas - vidas_actuales)
                        vidas_previas = vidas_actuales
                # --------------------------------------
                
                total_reward += reward
                steps += 1
                
                if USAR_SHIELD and 'safe_interventions' in info[0]:
                    intervenciones_finales = info[0]['safe_interventions']
                    
                if RENDERIZAR:
                    time.sleep(0.02) 

            # CÁLCULO DE MÉTRICAS FINALES
            eficiencia = total_reward[0] / steps if steps > 0 else 0
            ratio_seguridad = 100 * (1 - (intervenciones_finales / steps)) if steps > 0 else 0
            
            # Evitamos división por cero si no muere nunca (partida perfecta)
            puntos_por_muerte = (total_reward[0] / muertes) if muertes > 0 else float(total_reward[0])

            unsafe_ratio = 100 * (unsafe_steps / steps) if steps > 0 else 0


            print(f" 🔹 Episodio {i+1}: Pasos = {steps} | Puntos={total_reward[0]:.0f} | Muertes={muertes} | PPM={puntos_por_muerte:.2f} | Unsafe_ratio%={unsafe_ratio:.1f}% | Win={win} | Intervenciones={intervenciones_finales} | Ratio_Intervenciones: {ratio_seguridad}")
            
            resultados.append({
                "Episodio": i+1,
                "Pasos totales": steps, 
                "Recompensa": float(total_reward[0]),
                "Duracion": steps,
                "Intervenciones": intervenciones_finales,
                "Eficiencia_Pto_Paso": eficiencia,    
                "Ratio_Seguridad_IA": ratio_seguridad, 
                "Muertes": int(muertes),
                "Puntos_por_muerte": float(puntos_por_muerte),
                "Win": int(win),
                "EndReason": end_reason,
                "UnsafeSteps": int(unsafe_steps),
                "UnsafeRatio": float(unsafe_ratio),
                "Con_Escudo": USAR_SHIELD,
                "Distancia_Recompensa": DISTANCIA_RECOMPENSA,
                "Distancia_Escudo": DISTANCIA_ESCUDO if USAR_SHIELD else 0, # Guardamos la distancia usada
                "Modelo": MODELO_A_CARGAR
            })

    except KeyboardInterrupt:
        print("\n🛑 Validación detenida manualmente.")

    val_env.close()

    if resultados:
        df = pd.DataFrame(resultados)
        print("\n📊 --- RESUMEN DE VALIDACIÓN ---")
        print(f"Media de pasos por episodio: {df['Pasos totales'].mean():.2f}")
        print(f"Media de Puntos: {df['Recompensa'].mean():.2f}")
        print(f"Media de Intervenciones: {df['Intervenciones'].mean():.2f}")
        print(f"Media de Eficiencia: {df['Eficiencia_Pto_Paso'].mean():.3f} pts/paso")
        print(f"Seguridad Media (IA): {df['Ratio_Seguridad_IA'].mean():.1f}%")
        print(f"Media de Muertes: {df['Muertes'].mean():.2f}")
        print(f"Media Puntos/Muerte: {df['Puntos_por_muerte'].mean():.2f}")
        print(f"Media UnsafeSteps: {df['UnsafeSteps'].mean():.2f}")
        print(f"Media UnsafeRatio: {df['UnsafeRatio'].mean():.2f}%")
        print(f"Winrate: {100 * df['Win'].mean():.1f}%")
        
        # Nombre del archivo CSV actualizado con la distancia
        escudo_tag = f"_shield_d{DISTANCIA_ESCUDO}" if USAR_SHIELD else "_NoShield"
        reward_tag = f"_reward_d{DISTANCIA_RECOMPENSA}"
        nombre_csv = f"validacion_{MODELO_A_CARGAR}{escudo_tag}{reward_tag}.csv"
        ruta_completa = os.path.join(CARPETA_SALIDA, nombre_csv)
        
        # --- Añadir fila de medias ---
        cols_mean = [
            "Pasos totales", "Recompensa", "Duracion", "Intervenciones",
            "Eficiencia_Pto_Paso", "Ratio_Seguridad_IA", "Muertes",
            "Puntos_por_muerte", "Win", "UnsafeSteps", "UnsafeRatio"
        ]

        resumen = {c: "" for c in df.columns}   # por defecto vacío en todo
        resumen["Episodio"] = "MEDIA"

        for c in cols_mean:
            if c in df.columns:
                resumen[c] = df[c].mean()

        # Mantén metadata (opcional, pero queda bien)
        resumen["Con_Escudo"] = USAR_SHIELD
        resumen["Distancia_Recompensa"] = DISTANCIA_RECOMPENSA
        resumen["Distancia_Escudo"] = DISTANCIA_ESCUDO if USAR_SHIELD else 0
        resumen["Modelo"] = MODELO_A_CARGAR
        resumen["EndReason"] = ""

        df_out = pd.concat([df, pd.DataFrame([resumen])], ignore_index=True)

        df_out.to_csv(ruta_completa, index=False)

        print(f"📝 Resultados guardados en: {ruta_completa}")
    else:
        print("No se completó ningún episodio.")
