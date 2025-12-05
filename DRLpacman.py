import gymnasium as gym
import ale_py 

# Forzamos el registro de los entornos de Atari
gym.register_envs(ale_py)

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env_id = "ALE/MsPacman-v5"

# 1. Crear Entorno Vectorizado
print("2. Creando entorno vectorizado...")
# Si te sigue dando problemas de memoria, cambia n_envs=4 a n_envs=1
vec_env = make_atari_env(env_id, n_envs=4, seed=0)

# 2. Apilar Frames
env = VecFrameStack(vec_env, n_stack=4)

# 3. Crear Modelo (Optimizado para PC normal)
print("3. Inicializando modelo DQN con poca memoria...")
model = DQN("CnnPolicy", 
            env, 
            verbose=1,
            buffer_size=10000       # Reducimos el buffer de 1 millón a 10k
           )

# 4. Entrenar
print("4. Empezando el entrenamiento...")
model.learn(total_timesteps=10000, progress_bar=True)

# 5. Finalizar
print("✅ Entrenamiento finalizado.")
env.close()

# -----------------------------------------------------------
# 📺 PARTE VISUAL: ¡VER AL AGENTE JUGAR!
# -----------------------------------------------------------
import time

print("\n---------------------------------------")
print("🍿 ¡Preparando visualización! (Mira la nueva ventana)")
print("---------------------------------------")

# 1. Crear un entorno NUEVO específico para 'ver' (render_mode='human')
#    Usamos n_envs=1 porque solo queremos ver una pantalla.
#    Pasamos render_mode='human' dentro de env_kwargs.
visual_env = make_atari_env(env_id, n_envs=1, seed=0, 
                            env_kwargs={"render_mode": "human"})

# 2. IMPORTANTE: Aplicar el mismo 'stack' de frames que en el entrenamiento
#    Si no hacemos esto, el modelo no entenderá la entrada (espera 4 frames, recibiría 1).
visual_env = VecFrameStack(visual_env, n_stack=4)

# 3. Bucle de juego
obs = visual_env.reset()
done = False

try:
    # Ejecutamos hasta que termine el episodio o cierres la ventana
    while True:
        # El modelo predice la acción (deterministic=True es mejor para demos)
        action, _ = model.predict(obs, deterministic=True)
        
        # Ejecutar paso
        obs, reward, done, info = visual_env.step(action)
        
        # Pequeña pausa para que no vaya a velocidad de la luz y puedas verlo bien
        time.sleep(0.05) 
        
        # En entornos vectorizados, 'done' es un array de booleanos.
        # Reinician automáticamente, así que esto es un bucle infinito de partidas.
        # Si quieres ver solo una partida, podrías controlar las vidas en 'info'.

except KeyboardInterrupt:
    print("Visualización detenida por el usuario.")

visual_env.close()
print("👋 Fin del programa.")