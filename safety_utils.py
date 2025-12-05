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
        try:
            # Intento genérico para acceder a la RAM
            if hasattr(env, "unwrapped"):
                ram = env.unwrapped.ale.getRAM()
            else:
                # Si es un VecEnv, intentamos acceder al primer entorno
                ram = env.envs[0].unwrapped.ale.getRAM()
        except:
            # Fallback si no podemos leer RAM (devuelve ceros para no romper)
            return (0, 0), []

        # Posición Pacman
        pacman_pos = (ram[self.RAM_PACMAN_X], ram[self.RAM_PACMAN_Y])
        
        # Posición Fantasmas (lista de tuplas)
        ghosts_pos = []
        for i in range(4):
            g_pos = (ram[self.RAM_GHOSTS_X[i]], ram[self.RAM_GHOSTS_Y[i]])
            ghosts_pos.append(g_pos)
            
        return pacman_pos, ghosts_pos

    def is_danger(self, pacman_pos, ghosts_pos, threshold=30):
        """
        Verifica si algún fantasma está más cerca de 'threshold' (distancia Manhattan).
        threshold=30 es un valor aprox de 'peligro cercano'.
        """
        px, py = pacman_pos
        
        for gx, gy in ghosts_pos:
            # Distancia Manhattan: |x1 - x2| + |y1 - y2|
            dist = abs(px - gx) + abs(py - gy)
            
            if dist < threshold:
                return True, dist # ¡PELIGRO!
                
        return False, 999 # A salvo

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