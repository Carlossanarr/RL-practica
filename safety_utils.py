import numpy as np

class PacmanSafetyMonitor:
    def __init__(self):
        # Offsets de memoria RAM para Ms. Pac-Man (Atari 2600)
        # Estos son los bytes específicos donde la consola guarda las posiciones
        self.RAM_PACMAN_X = 10
        self.RAM_PACMAN_Y = 16
        
        # Hay 4 fantasmas. X en 6-9, Y en 12-15
        self.RAM_GHOSTS_X = [6, 7, 8, 9]
        self.RAM_GHOSTS_Y = [12, 13, 14, 15]

    def get_positions(self, env):
        """
        Extrae las coordenadas (x, y) de Pacman y los Fantasmas leyendo la RAM.
        NOTA: Requiere acceso al entorno base de ALE.
        """
        # Desempaquetamos el entorno para llegar al nucleo de Atari (ALE)
        # Dependiendo de los wrappers, puede estar en .unwrapped o .env
        # Intentamos acceder a la RAM del entorno base ALE
        try:
            # Caso normal: entorno no vectorizado
            if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "ale"):
                ram = env.unwrapped.ale.getRAM()

            # Caso VecEnv: acceder al primer entorno
            elif hasattr(env, "envs") and len(env.envs) > 0:
                base_env = env.envs[0]
                if hasattr(base_env, "unwrapped") and hasattr(base_env.unwrapped, "ale"):
                    ram = base_env.unwrapped.ale.getRAM()
                else:
                    raise RuntimeError("Env vectorizado sin acceso a unwrapped.ale")

            else:
                raise RuntimeError("No se pudo encontrar un entorno ALE válido")

        except Exception as e:
            raise RuntimeError(
                f"FALLO CRÍTICO: no se pudo leer la RAM de ALE ({e}). "
                "Revisa la cadena de wrappers o el env pasado a get_positions()."
            )

        # Posición Pacman
        pacman_pos = (int(ram[self.RAM_PACMAN_X]), int(ram[self.RAM_PACMAN_Y]))
        
        # Posición Fantasmas (lista de tuplas)
        ghosts_pos = []
        for i in range(4):
            g_pos = (int(ram[self.RAM_GHOSTS_X[i]]), int(ram[self.RAM_GHOSTS_Y[i]]))
            ghosts_pos.append(g_pos)
            
        return pacman_pos, ghosts_pos

    def is_danger(self, pacman_pos, ghosts_pos, threshold=30):
        """
        Devuelve:
        - is_danger (bool): True si el fantasma más cercano está dentro del umbral
        - min_dist (int): distancia Manhattan mínima al fantasma más cercano
        """

        # Seguridad extra: si no hay fantasmas, no hay peligro
        if not ghosts_pos:
            return False, 999

        px, py = pacman_pos

        # Distancia mínima real
        min_dist = min(
            abs(px - gx) + abs(py - gy)
            for gx, gy in ghosts_pos
        )

        is_unsafe = min_dist < threshold

        return is_unsafe, min_dist


    def get_safe_action(self, pacman_pos, ghosts_pos):
        """
        Lógica simple del 'Teacher': Si hay peligro, muévete al lado contrario.
        Esto es una heurística básica.
        """
        # Acciones en Atari: 0:NOOP, 1:UP, 2:RIGHT, 3:LEFT, 4:DOWN (aprox, depende de la versión)
        # Vamos a simplificar: huir del fantasma más cercano.
        
        px, py = pacman_pos
        # Buscar fantasma más cercano
        closest_g = min(ghosts_pos, key=lambda g: abs(px - g[0]) + abs(py - g[1]))
        gx, gy = closest_g
        
        # Lógica de huida (invertir dirección)
        if abs(px - gx) > abs(py - gy):
            # Diferencia mayor en horizontal, huimos horizontalmente
            if px < gx: return 3 # LEFT (Huir a la izq si fantasma está a la der)
            else: return 2 # RIGHT
        else:
            # Huimos verticalmente
            if py < gy: return 1 # UP
            else: return 4 # DOWN
