#!/usr/bin/env python3
"""
Aplicación de aleatorización WRO Future Engineers

Esta aplicación aleatoriza la posición de inicio del vehículo, la configuración
de paredes interiores y posiciones de obstáculos para WRO Future Engineers.

Es una aplicación Flask que genera imágenes del campo de juego con elementos
aleatorizados.

El tipo de desafío y la dirección de conducción del desafío se
definen a través de los puntos finales de la URL:

El tipo de desafío y la dirección de conducción del desafío se
definen a través de los puntos finales de la URL:

- `/qualification/cw` - el desafío abierto ronda con conducción en el sentido
de las agujas del reloj.

- `/qualification/ccw` - el desafío abierto ronda con conducción en sentido
antihorario.

- `/final/cw` - El desafío de obstáculos alrededor con la dirección de
conducción en el sentido de las agujas del reloj.

- `/final/ccw` - El desafío de obstáculos ronda con conducción en sentido
antihorario.
"""

from flask import Flask, make_response, render_template
import cv2
import numpy as np
from random import randint, choice, sample
from enum import Enum
import random

app = Flask(__name__)
app.debug = True

"""
### Descripción general del campo de juego

El campo de juego es un cuadrado con dimensiones de 3 metros por 3 metros,
dividido en 9 secciones iguales, cada una de 1 metro de tamaño por 1 metro.
Las secciones están organizadas de la siguiente manera:

1. **Secciones de esquina**:
    - Hay 4 secciones de esquina, ubicadas en las cuatro esquinas exteriores
    de la alfombra. Estas secciones se denominan "C" en el esquema a
    continuación.

2. **Secciones rectas**:
    - Hay 4 secciones rectas, que son lineales y conectan esquinas opuestas.
    Estas secciones están posicionadas como:
        - 2 secciones rectas horizontales (una arriba y otra abajo de
        la sección central).
        - 2 secciones rectas verticales (una a la izquierda y otra a la
        derecha de la sección central).
    - Estas secciones se denominan "S" en el esquema.

3. **Sección central**:
    - La sección del medio es no funcional y sirve como la **sección central**.
      Esta área se denomina "###" en el esquema
    y permanece sin usar durante el juego.

**Representación pseudográfica del campo de juego**

Para ayudar a visualmente entender el diseño, aquí tienes una representación
pseudográfica del campo de juego:


```
+---+---+---+
|   |   |   |
| C | S | C |
|   |   |   |
+---+---+---+
| S |###| S |
|   |###|   |  Sección central (###)
|   |###|   |  (Sin uso durante el juego)
+---+---+---+
|   |   |   |
| C | S | C |
|   |   |   |
+---+---+---+
```

**Notas importantes:**

- El campo de juego consiste únicamente en las secciones descritas
anteriormente, sin paredes o obstáculos incorporados.

- **Paredes**: Las paredes exteriores e interiores son elementos de juego
adicionales que se colocan alrededor o sobre la campo de juego durante el
juego, pero no son parte de la estructura del campo de juego en sí.

- Las **paredes exteriores** generalmente corren a lo largo de los bordes de
del campo, mientras que las **paredes interiores** se pueden configurar de
varias maneras, potencialmente encierran la sección central, pero estas paredes
son separadas de la estructura de la campo de juego.

**Etiquetas para las secciones rectas**

- La sección en la parte superior está etiquetada como "Sección N"
- La sección en la parte inferior está etiquetada como "Sección S"
- La sección a la izquierda está etiquetada como "Sección W"
- La sección a la derecha está etiquetada como "Sección E"

```
+---+---+---+
|   |   |   |
|   | N |   |
|   |   |   |
+---+---+---+
| W |###| E |
|   |###|   |
|   |###|   |
+---+---+---+
|   |   |   |
|   | S |   |
|   |   |   |
+---+---+---+
```

### Descripción de la disposición de la sección recta

Cada sección recta en el campo de juego tiene un diseño estructurado,
que presenta los siguientes elementos:

1. **Radios**:
    - Hay **tres radios** dentro de cada sección recta:

        - **Dos radios** corren verticalmente a lo largo de los **bordes
        izquierdo y derecho** de la sección. Se extienden desde la sección
        central (parte superior) hacia la parte exterior del campo (parte
        inferior).

        - **Un radio central** está posicionado en el **medio** de la sección,
        corriendo verticalmente desde la parte superior hasta la parte
        inferior de la sección.

    - Estos radios dividen la sección en dos columnas de zonas.

2. **Arcos**:
    - Hay **dos arcos** que corren horizontalmente a través de la sección:
        - Los arcos están posicionados a dos niveles, creando divisiones a
        través de la sección horizontalmente.
        - Estos arcos no conectan las secciones de esquina, sino que dividen
        la sección en múltiples zonas horizontalmente.

3. **Intersecciones**:
    - El diseño de radios y arcos forma múltiples intersecciones, que pueden
    ser utilizadas para colocar obstáculos como señales de tránsito:
        - **4 intersecciones en T**: Estas ocurren donde los arcos se
        encuentran con los radios.
        - **2 intersecciones en X**: Estas ocurren donde el radio central se
        intersecta con los arcos.
    - Cada intersección es una ubicación potencial para colocar obstáculos que
    los vehículos deben navegar durante el juego.

4. **Zonas**:
    - La combinación de los dos arcos y tres radios divide la sección recta en
        **6 zonas**:
        - Las **zonas superior e inferior** son más altas y ofrecen más
        espacio.
        - Las **zonas medias** son más cortas, ya que están comprimidas por
        los arcos.
    - Estas zonas sirven como áreas clave para el juego, donde se pueden
    colocar obstáculos o pueden ocurrir acciones específicas.

**Representación pseudográfica de la sección recta:**

```
radio   radio
v       v
+---+---+
|   |   |
|   |   |
|   |   |
+---+---+ - arco
|   |   |
|   |   |
+---+---+ - arco
|   |   |
|   |   |
|   |   |
+---+---+
    ^
    radio
```

En este esquema:
- Los tres radios (izquierdo, derecho y medio) corren verticalmente desde la
parte exterior del campo de juego (en la parte superior) hasta la sección
central (en la parte inferior).

- Los dos arcos dividen la sección horizontalmente en tres filas de zonas,
siendo la fila del medio más corta que las filas superior e inferior.

- Las intersecciones formadas por los radios y arcos sirven como ubicaciones
clave para la colocación de obstáculos.

**Etiquetas para los puntos de intersección**

- Los puntos de intersección en la fila superior están etiquetados
como "T4", "X2" y "T3" de izquierda a derecha.

- Los puntos de intersección en la fila inferior están etiquetados
como "T2", "X1" y "T1" de izquierda a derecha.

**Etiquetas para las zonas**

- Las zonas en la fila superior están etiquetadas como "Z6" y "Z5"
- Las zonas en la fila del medio están etiquetadas como "Z4" y "Z3"
- Las zonas en la fila inferior están etiquetadas como "Z2" y "Z1"


```
+----+----+
|    |    |
| Z6 | Z5 |
|    |    |
T4---X2---T3
| Z4 | Z3 |
|    |    |
T2---X1---T1
|    |    |
| Z2 | Z1 |
|    |    |
+----+----+
```
"""

# coordenadas

# Para poder aleatorizar la configuración de las paredes interiores
# así como las posiciones de los obstáculos, es necesario definir
# los puntos principales del campo de juego en píxeles.

# El tamaño del campo de juego es de 3 metros por 3 metros,
# que son 3000 píxeles por 3000 píxeles.
# El ancho y la altura no incluyen las paredes exteriores.
width = 3000
height = width

# La sección recta es de 1000x1000 píxeles.
# El campo tiene 4 secciones rectas con el mismo diseño.
# Debido a la simetría, es suficiente definir las coordenadas de
# los elementos para una sección relativa a la esquina superior izquierda
# de la alfombra (0,0).
# Las coordenadas a continuación son para la sección etiquetada "Sección N".

# Arcos:
# La primera línea es el primer arco en la sección recta.
first_line = 400
# La segunda línea es el segundo arco en la sección recta.
second_line = first_line + 200

# La frontera interior es la frontera entre la sección recta y la
# sección central.
inner_border = second_line + 400

# Radios:
# La posición izquierda es el radio izquierdo en la sección recta.
left_position = inner_border
# La posición central es el radio central en la sección recta.
middle_position = width // 2
# La posición derecha es el radio derecho en la sección recta.
right_position = width - inner_border

# Grosor de las líneas que representan las paredes
border = 10

# Grosor de las líneas que representan los radios y arcos
thin_line = 2

# La presentación de la dirección de conducción del desafío es un arco estrecho
# en la sección central de la alfombra de juego.
narrow_radius = 350
narrow_color = (255, 0, 0)  # El color es azul
narrow_thickness = 20

# El color para marcar la zona de inicio es gris
start_section_color = (192, 192, 192)

# El color de las barreras del estacionamiento es magenta
parking_lot_color = (255, 0, 255)
parking_barrier_thickness = 20
parking_barrier_length = 200
distance_between_parking_barriers = 300

# El obstáculo será representado como un cuadrado con un lado de 100 píxeles.
obstacle_size = 100


def on_north(img, h1, w1, h2, w2, c):
    """
    Dibuja un cuadrado con el color dado y las coordenadas relativas
    en la Sección N.
    """

    img[
        min(h1, h2) + border:max(h1, h2)+border,
        min(w1, w2) + border:max(w1, w2)+border
        ] = c


def on_south(img, h1, w1, h2, w2, c):
    """
    Draw a square with the given color and the given relative coordinates
    in the Section S
    """

    img[
        height-max(h1, h2) + border:height-min(h1, h2)+border,
        width-max(w1, w2) + border:width-min(w1, w2)+border
        ] = c


def on_west(img, h1, w1, h2, w2, c):
    """
    Dibuja un cuadrado con el color dado y las coordenadas relativas
    en la Sección W.
    """

    img[
        height-max(w1, w2)+border:height-min(w1, w2) + border,
        min(h1, h2)+border:max(h1, h2) + border
        ] = c


def on_east(img, h1, w1, h2, w2, c):
    """
    Dibuja un cuadrado con el color dado y las coordenadas relativas
    en la Sección E.
    """

    img[
        min(w1, w2) + border:max(w1, w2) + border,
        width-max(h1, h2)+border:width-min(h1, h2)+border
        ] = c


class Section(Enum):
    """
    Representa una sección recta.

    El valor es una función para dibujar un cuadrado en la
    sección recta correspondiente.
    """
    NORTH = on_north
    SOUTH = on_south
    WEST = on_west
    EAST = on_east


class Direction(Enum):
    CW = 'cw'
    CCW = 'ccw'

    @classmethod
    def is_cw(cls, direction):
        return direction == cls.CW

    @classmethod
    def is_ccw(cls, direction):
        return direction == cls.CCW


class ChallengeType(Enum):
    OPEN = 'open'
    OBSTACLE = 'obstacle'


# Puntos de intersección en las secciones rectas
class Intersection(Enum):
    TopLeft = (0, left_position)
    TopMiddle = (0, middle_position)
    TopRight = (0, right_position)
    T4 = (first_line, left_position)
    X2 = (first_line, middle_position)
    T3 = (first_line, right_position)
    T2 = (second_line, left_position)
    X1 = (second_line, middle_position)
    T1 = (second_line, right_position)
    BottomLeft = (inner_border, left_position)
    BottomMiddle = (inner_border, middle_position)
    BottomRight = (inner_border, right_position)


class Color(Enum):
    RED = (55, 39, 238)
    GREEN = (44, 214, 68)
    UNDEFINED = (0, 0, 0)


class Obstacle:
    """
    Representa un obstáculo en la alfombra de juego.

    El obstáculo se define por la posición y el color.
    """

    def __init__(self, position: Intersection, color: Color):
        self.position = position
        self.color = color

    def set_color(self, color: Color):
        self.color = color

    def is_red(self):
        return self.color == Color.RED

    def is_green(self):
        return self.color == Color.GREEN

    def _x(self):
        return self.position.value[0]

    def _y(self):
        return self.position.value[1]

    def _color(self):
        return self.color.value

    def draw(self, img: np.ndarray, section: Section):
        """
        Dibuja un obstáculo cuadrado en la sección recta definida por la
        función `section` dada.
        """

        section(img,
                self._x()-(obstacle_size//2), self._y()-(obstacle_size//2),
                self._x()+(obstacle_size//2), self._y()+(obstacle_size//2),
                self._color())


class StartZone(Enum):
    Z1 = (Intersection.X1, Intersection.BottomRight)
    Z2 = (Intersection.T2, Intersection.BottomMiddle)
    Z3 = (Intersection.X2, Intersection.T1)
    Z4 = (Intersection.T4, Intersection.X1)
    Z5 = (Intersection.TopMiddle, Intersection.T3)
    Z6 = (Intersection.TopLeft, Intersection.X2)

# Intersecciones que estarán frente al vehículo para la dirección dada
# y la zona de inicio dada.
#
# Dirección en el sentido horario:
#   Z3: T1, T3
#   Z4: X1, X2
#
# Dirección en sentido antihorario:
#   Z3: X1, X2
#   Z4: T2, T4


forbidden_intersections_in_start_zone = {
    Direction.CW: {
        StartZone.Z3: [Intersection.T1, Intersection.T3],
        StartZone.Z4: [Intersection.X1, Intersection.X2]
    },
    Direction.CCW: {
        StartZone.Z3: [Intersection.X1, Intersection.X2],
        StartZone.Z4: [Intersection.T2, Intersection.T4]
    }
}

# Según las reglas, estas intersecciones en la sección recta que contiene el
# estacionamiento no pueden ser utilizadas para la colocación de obstáculos.
forbidden_intersections_in_parking_section = [
    Intersection.T3,
    Intersection.T4,
    Intersection.X2
]


class VehiclePosition:
    """
     Representa la posición de inicio de un vehículo en el campo de juego.
    """

    def __init__(self, start_zone: StartZone):
        self.start_zone = start_zone

    def _top_left_x(self):
        return self.start_zone.value[0].value[0]

    def _top_left_y(self):
        return self.start_zone.value[0].value[1]

    def _bottom_right_x(self):
        return self.start_zone.value[1].value[0]

    def _bottom_right_y(self):
        return self.start_zone.value[1].value[1]

    def draw(self, img: np.ndarray, section: Section):
        """
        Dibuja una zona de inicio de vehículo en la sección recta definida
        por la función `section` dada.
        """

        section(img,
                self._top_left_x(), self._top_left_y(),
                self._bottom_right_x(), self._bottom_right_y(),
                start_section_color)


class InnerWall:
    """
    Representa las paredes internas del campo de juego.
    """

    def __init__(self, sides: list[Section] = []):
        self._north = Section.NORTH in sides
        self._west = Section.WEST in sides
        self._south = Section.SOUTH in sides
        self._east = Section.EAST in sides

        # Inicializa las paredes en el medio como fijas
        self.fixed_center = True

    def on_north(self):
        """
        Comprueba si la pared interior está más cerca de la pared exterior en
        el lado norte de la alfombra de juego.
        """
        return self._north

    def on_west(self):
        """
        Comprueba si la pared interior está más cerca de la pared exterior en
        el lado oeste de la alfombra de juego.
        """
        return self._west

    def on_south(self):
        """
        Comprueba si la pared interior está más cerca de la pared exterior en
        el lado sur de la alfombra de juego.
        """
        return self._south

    def on_east(self):
        """
        Comprueba si la pared interior está más cerca de la pared exterior en
        el lado este de la alfombra de juego.
        """
        return self._east

    def on_side(self, side: Section):
        """
        Comprueba si la pared interior está más cerca de la pared exterior en
        el lado dado de la alfombra de juego.
        """
        if side == Section.NORTH:
            return self.on_north()
        elif side == Section.WEST:
            return self.on_west()
        elif side == Section.SOUTH:
            return self.on_south()
        elif side == Section.EAST:
            return self.on_east()

    def draw(self, img: np.ndarray):
        """
        Dibuja las paredes interiores de la alfombra de juego.
        """

        # posición predeterminada de las paredes interiores
        h_n = inner_border           # coordenada Y de la pared interior norte
        w_w = inner_border           # coordenada X de la pared interior oeste
        h_s = height - inner_border  # coordenada Y de la pared interior sur
        w_e = width - inner_border   # coordenada X de la pared interior este

        # Ajusta la posición de las paredes interiores en función de qué lado
        # de la alfombra la pared interior debe ser dibujada más cerca de las
        # paredes exteriores - la pared se posiciona a lo largo del segundo
        # arco de la sección recta correspondiente.

        if self.on_north():
            h_n = second_line
        if self.on_west():
            w_w = second_line
        if self.on_south():
            h_s = height - second_line
        if self.on_east():
            w_e = width - second_line

        # norte
        img[
            h_n - (border//2) + border:h_n + (border//2) + border,
            w_w - (border//2) + border:w_e + (border//2) + border
            ] = (0, 0, 0)
        # oeste
        img[
            h_n - (border//2) + border:h_s + (border//2) + border,
            w_w - (border//2) + border:w_w + (border//2) + border
            ] = (0, 0, 0)
        # sur
        img[
            h_s - (border//2) + border:h_s + (border//2) + border,
            w_w - (border//2) + border:w_e + (border//2) + border
            ] = (0, 0, 0)
        # este
        img[
            h_n - (border//2) + border:h_s + (border//2) + border,
            w_e - (border//2) + border:w_e + (border//2) + border
            ] = (0, 0, 0)

# El proceso de aleatorización opera con conjuntos de obstáculos.
# Cada elemento de la lista define posiciones relativas de los obstáculos en la
# sección recta.
# Se eliminan duplicados para las Tarjetas (Card) 14, 15, 20, 21 para tener
# resultados más valiosos desde el punto de vista de evaluación.


obstacles_sets = [
    # Obstáculos de intersección única (T1)
    [Obstacle(Intersection.T1, Color.GREEN)],                # 0, Card 1
    [Obstacle(Intersection.T1, Color.RED)],                  # 1, Card 2

    # Obstáculos de intersección única (X1)
    [Obstacle(Intersection.X1, Color.GREEN)],                # 2, Card 3
    [Obstacle(Intersection.X1, Color.RED)],                  # 3, Card 4

    # Obstáculos de intersección única (T2)
    [Obstacle(Intersection.T2, Color.GREEN)],                # 4, Card 5
    [Obstacle(Intersection.T2, Color.RED)],                  # 5, Card 6

    # Obstáculos de intersección única (T3)
    [Obstacle(Intersection.T3, Color.GREEN)],                # 6, Card 7
    [Obstacle(Intersection.T3, Color.RED)],                  # 7, Card 8

    # Obstáculos de intersección única (X2)
    [Obstacle(Intersection.X2, Color.GREEN)],                # 8, Card 9
    [Obstacle(Intersection.X2, Color.RED)],                  # 9, Card 10

    # Obstáculos de intersección única (T4)
    [Obstacle(Intersection.T4, Color.GREEN)],                # 10, Card 11
    [Obstacle(Intersection.T4, Color.RED)],                  # 11, Card 12

    # Combinaciones de T3 y T2
    [Obstacle(Intersection.T3, Color.GREEN),
     Obstacle(Intersection.T2, Color.GREEN)],                # 12, Card 13
    [Obstacle(Intersection.T3, Color.GREEN),
     Obstacle(Intersection.T2, Color.RED)],                  # 13, Card 14/16
    [Obstacle(Intersection.T3, Color.RED),
     Obstacle(Intersection.T2, Color.GREEN)],                # 14, Card 15/17
    [Obstacle(Intersection.T3, Color.RED),
     Obstacle(Intersection.T2, Color.RED)],                  # 15, Card 18

    # Combinaciones de T1 y T4
    [Obstacle(Intersection.T1, Color.GREEN),
     Obstacle(Intersection.T4, Color.GREEN)],                # 16, Card 19
    [Obstacle(Intersection.T1, Color.GREEN),
     Obstacle(Intersection.T4, Color.RED)],                  # 17, Card 20/22
    [Obstacle(Intersection.T1, Color.RED),
     Obstacle(Intersection.T4, Color.GREEN)],                # 18, Card 21/23
    [Obstacle(Intersection.T1, Color.RED),
     Obstacle(Intersection.T4, Color.RED)],                  # 19, Card 24

    # Combinaciones de T1 y T2
    [Obstacle(Intersection.T1, Color.GREEN),
     Obstacle(Intersection.T2, Color.GREEN)],                # 20, Card 25
    [Obstacle(Intersection.T1, Color.GREEN),
     Obstacle(Intersection.T2, Color.RED)],                  # 21, Card 26
    [Obstacle(Intersection.T1, Color.RED),
     Obstacle(Intersection.T2, Color.GREEN)],                # 22, Card 27
    [Obstacle(Intersection.T1, Color.GREEN),
     Obstacle(Intersection.T2, Color.RED)],                  # 23, Card 28
    [Obstacle(Intersection.T1, Color.RED),
     Obstacle(Intersection.T2, Color.GREEN)],                # 24, Card 29
    [Obstacle(Intersection.T1, Color.RED),
     Obstacle(Intersection.T2, Color.RED)],                  # 25, Card 30

    # Combinaciones de T3 y T4
    [Obstacle(Intersection.T3, Color.GREEN),
     Obstacle(Intersection.T4, Color.GREEN)],                # 26, Card 31
    [Obstacle(Intersection.T3, Color.GREEN),
     Obstacle(Intersection.T4, Color.RED)],                  # 27, Card 32
    [Obstacle(Intersection.T3, Color.RED),
     Obstacle(Intersection.T4, Color.GREEN)],                # 28, Card 33
    [Obstacle(Intersection.T3, Color.GREEN),
     Obstacle(Intersection.T4, Color.RED)],                  # 29, Card 34
    [Obstacle(Intersection.T3, Color.RED),
     Obstacle(Intersection.T4, Color.GREEN)],                # 30, Card 35
    [Obstacle(Intersection.T3, Color.RED),
     Obstacle(Intersection.T4, Color.RED)],                  # 31, Card 36
]

# El proceso de aleatorización dice que al menos uno de los
# secciones rectas debe tener al menos un obstáculo
# en la intersección etiquetada como "X2". El mapa contiene índices
# de los correspondientes conjuntos de obstáculos para
# evitar riesgos cuando las soluciones incompletas resuelven el desafío.

mandatory_obstacles_sets = {
    Color.GREEN: 8,
    Color.RED: 9
}

# Uno de los conjuntos de obstáculos de esta lista debe estar presente en el
# campo de juego para reducir el riesgo cuando soluciones incompletas
# resuelven el desafío.

required_obstacles_sets = [21, 22, 27, 28]

# Posiciones relativas de las zonas de inicio del vehículo en las
# secciones rectas para las rondas de desafío abiertas.

vehicle_positions_in_open = [
    VehiclePosition(StartZone.Z6),
    VehiclePosition(StartZone.Z5),
    VehiclePosition(StartZone.Z4),
    VehiclePosition(StartZone.Z3),
    VehiclePosition(StartZone.Z2),
    VehiclePosition(StartZone.Z1)
]

# Posiciones relativas de las zonas de inicio del vehículo en las
# secciones rectas para las rondas de desafío de obstáculos.

vehicle_positions_in_obstacle = [
    VehiclePosition(StartZone.Z4),
    VehiclePosition(StartZone.Z3)
]

# Plantilla de imagen del campo de juego
template = np.zeros((height+border * 2,
                     width+border * 2, 3), np.uint8)

# El campo de juego es un cuadrado blanco con el borde que
# representa las paredes exteriores.
# El color del borde es negro.

template[
    border:height + border,
    border:width+border
    ] = (255, 255, 255)

# Las líneas que representan arcos en la "Sección N"
template[
    first_line - (thin_line//2) + border:
    first_line + (thin_line//2) + border,
    border + inner_border:width - inner_border + border
    ] = (0, 0, 0)

template[
    second_line - (thin_line//2) + border:
    second_line + (thin_line//2) + border,
    border + inner_border: width - inner_border + border
    ] = (0, 0, 0)

# Una línea que representa el radio izquierdo de la "Sección W", el borde de la
# "Sección N" con la sección central y el radio derecho de la "Sección E".
template[
    inner_border - (thin_line//2) + border:
    inner_border + (thin_line//2) + border,
    border:width + border
    ] = (0, 0, 0)

# Las líneas que representan arcos en la "Sección S"
template[
    height - first_line - (thin_line//2) + border:
    height - first_line + (thin_line//2) + border,
    border + inner_border: width - inner_border + border
    ] = (0, 0, 0)

template[
    height - second_line - (thin_line//2) + border:
    height - second_line + (thin_line//2) + border,
    border + inner_border: width - inner_border + border
    ] = (0, 0, 0)

# Una línea que representa el radio derecho de la "Sección W", el borde de la
# "Sección S" con la sección central y el radio izquierdo de la "Sección E".
template[
    height - inner_border - (thin_line//2) + border:
    height - inner_border + (thin_line//2) + border,
    border: width + border
    ] = (0, 0, 0)

# Las líneas que representan arcos en la "Sección W"
template[
    border+inner_border:
    height-inner_border+border,
    first_line-(thin_line//2)+border:
    first_line+(thin_line//2)+border
    ] = (0, 0, 0)
template[
    border + inner_border:
    height - inner_border + border,
    second_line - (thin_line//2) + border:
    second_line + (thin_line//2) + border
    ] = (0, 0, 0)

# Una línea que representa el radio izquierdo de la "Sección N", el borde de la
# "Sección E" con la sección central y el radio derecho de la "Sección S".
template[
    border:height + border,
    inner_border - (thin_line//2) + border:
    inner_border + (thin_line//2) + border
    ] = (0, 0, 0)

# Las líneas que representan arcos en la "Sección E"
template[
    border + inner_border:
    height - inner_border+border,
    width - first_line - (thin_line//2) + border:
    width - first_line + (thin_line//2) + border
    ] = (0, 0, 0)

template[
    border + inner_border:
    height - inner_border+border,
    width - second_line - (thin_line//2) + border:
    width - second_line + (thin_line//2) + border
    ] = (0, 0, 0)

# Una línea que representa el radio derecho de la "Sección N", el borde de la
# "Sección W" con la sección central y el radio izquierdo de la "Sección S".
template[
    border: height + border,
    width - inner_border - (thin_line//2) + border:
    width - inner_border + (thin_line//2) + border
    ] = (0, 0, 0)

# La línea que representa el radio central de la "Sección N"
template[
    border: inner_border + border,
    (width//2) - (thin_line//2) + border:
    (width//2) + (thin_line//2) + border
    ] = (0, 0, 0)

# La línea que representa el radio central de la "Sección S"
template[
    height - inner_border + border:
    height + border,
    (width//2) - (thin_line//2) + border:
    (width//2) + (thin_line//2) + border
    ] = (0, 0, 0)

# La línea que representa el radio central de la "Sección W"
template[
    (height//2) - (thin_line//2) + border:
    (height//2) + (thin_line//2) + border,
    border: inner_border + border
    ] = (0, 0, 0)

# La línea que representa el radio central de la "Sección E"
template[
    (height//2) - (thin_line//2) + border:
    (height//2) + (thin_line//2) + border,
    width - inner_border + border: width + border
    ] = (0, 0, 0)


def draw_parking_lot_barriers(img, section: Section):
    """
    Dibuja las barreras del estacionamiento en la sección dada.
    """
    # coordenadas de la esquina superior izquierda de la primera barrera
    # relativamente a la sección:

    first_barrier_top_left = (
        left_position,  # coordenada x
        0               # coordenada y - alineada con el borde superior
        )

    # coordenadas de la esquina inferior derecha de la primera barrera
    # relativamente a la sección:

    first_barrier_bottom_right = (
        left_position + parking_barrier_thickness,  # coordenada x
        parking_barrier_length  # coordenada y - se extiende hacia abajo por
                                # la longitud de la barrera
        )

    # coordenadas de la esquina superior izquierda de la segunda barrera
    # relativamente a la sección:

    second_barrier_top_left = (
        left_position + parking_barrier_thickness +
        distance_between_parking_barriers,  # coordenada x
        0           # coordenada y - alineada con el borde superior
        )

    # coordenadas de la esquina inferior derecha de la segunda barrera
    # relativamente a la sección:

    second_barrier_bottom_right = (
        left_position +
        parking_barrier_thickness +
        distance_between_parking_barriers +
        parking_barrier_thickness,  # coordenada x
        parking_barrier_length  # coordenada y - se extiende hacia abajo por
                                # la longitud de la barrera
        )

    # Dibuja ambas barreras

    section(img,
            first_barrier_top_left[1],
            first_barrier_top_left[0],
            first_barrier_bottom_right[1],
            first_barrier_bottom_right[0],
            parking_lot_color
            )

    section(img,
            second_barrier_top_left[1],
            second_barrier_top_left[0],
            second_barrier_bottom_right[1],
            second_barrier_bottom_right[0],
            parking_lot_color
            )


def draw_obstacles_set(img, section: Section, obstacles_set: list[Obstacle]):
    """
    Dibuja un conjunto de obstáculos definidos por los elementos de
    `obstacles_set` en la sección recta definida por la función `section`.
    """

    for obstacle in obstacles_set:
        obstacle.draw(img, section)


def draw_narrow(img, direction: Direction):
    """
    Dibuja el arco estrecho en la sección central del campo de juego.
    `direction` contiene la dirección de conducción del vehículo.
    """

    # El centro del campo de juego se calcula teniendo en cuenta las paredes
    # exteriores.
    img_center = (width // 2 + border, height // 2 + border)

    axes = (narrow_radius, narrow_radius)
    startA = 180
    endA = -90
    # Dibuja el arco
    img = cv2.ellipse(img, img_center, axes, 0, startA, endA,
                      narrow_color, narrow_thickness)

    # Dibuja la flecha al final del arco
    if Direction.is_cw(direction):
        startP = (img_center[0] - narrow_radius, img_center[1])
        endP = (img_center[0] - narrow_radius - 30, img_center[1] + 80)
        img = cv2.line(img, startP, endP, narrow_color, narrow_thickness)
        endP = (img_center[0] - narrow_radius + 50, img_center[1] + 75)
        img = cv2.line(img, startP, endP, narrow_color, narrow_thickness)

    elif Direction.is_ccw(direction):
        startP = (img_center[0], img_center[1] - narrow_radius)
        endP = (img_center[0] + 80, img_center[1] - narrow_radius - 30)
        img = cv2.line(img, startP, endP, narrow_color, narrow_thickness)
        endP = (img_center[0] + 75, img_center[1] - narrow_radius + 50)
        img = cv2.line(img, startP, endP, narrow_color, narrow_thickness)


def draw_scheme_for_final(scheme):
    """
    Dibuja el campo de juego para las rondas de desafío de obstáculos.

    El esquema es un diccionario con las siguientes claves:

    - start_section: la sección recta donde se encuentra la zona de inicio

    - start_zone: la posición de la zona de inicio en la sección recta elegida

    - obstacles: un diccionario donde las claves son índices de los conjuntos
    de obstáculos y los valores son secciones donde se encuentran los
    obstáculos.

    - parking_section: la sección donde se encuentra el estacionamiento.

    Devuelve un arreglo tridimensional de NumPy (matriz) que representa el
    campo de juego
    donde cada píxel está representado por tres números que corresponden al
    color BGR.
    """

    image = template.copy()

    # Crea el objeto de posición de inicio del vehículo para la zona dada
    # y dibuja en la sección recta elegida
    VehiclePosition(scheme['start_zone']).draw(image, scheme['start_section'])

    # Dibuja las barreras del estacionamiento en la sección de estacionamiento
    draw_parking_lot_barriers(image, scheme['parking_section'])

    # Dibuja los obstáculos en las secciones correspondientes
    obstacles_configuration = scheme['obstacles']
    for obstacles_set_index in obstacles_configuration:
        draw_obstacles_set(
            image, obstacles_configuration[obstacles_set_index],
            obstacles_sets[obstacles_set_index]
            )

    return image


def draw_layout(direction: Direction, fixed: bool) -> np.ndarray:
    """Genera el campo de juego y dibuja sobre él."""

    # No se puede usar list(Section) porque los elementos de Section son
    # funciones.
    sections = [Section.NORTH, Section.WEST, Section.SOUTH, Section.EAST]

    # Elige en qué lados de la alfombra de juego las paredes interiores
    # deben dibujarse más cerca de las paredes exteriores.
    inner_walls_config = sample(sections, randint(0, 4))
    inner_walls = InnerWall(inner_walls_config)

    # Elige la sección recta donde se encuentra la zona de inicio.
    starting_section = choice(sections)

    # Si la pared interior en la sección de inicio está más cerca de la pared
    # exterior, la zona de inicio podría ser solo una de las cuatro zonas
    # disponibles en la sección de inicio. Por lo tanto, el número de zonas
    # utilizadas para la aleatorización debe ser limitado.
    allowed_zones = (
        [StartZone.Z6, StartZone.Z5, StartZone.Z4, StartZone.Z3] if
        inner_walls.on_side(starting_section)
        else list(StartZone)
    )

    # Elige la zona de inicio dentro de las zonas permitidas.
    starting_zone = choice(allowed_zones)

    image = template.copy()

    # Crea el objeto de posición de inicio del vehículo para la zona dada
    # y dibuja en la sección recta elegida
    VehiclePosition(starting_zone).draw(image, starting_section)

    # Dibuja las paredes dependiendo de si el centro es fijo o aleatorio
    if fixed:
        inner_walls = InnerWall()  # Usar un centro fijo
        inner_walls.draw(image)
    else:
        inner_walls.draw(image)

    # Dibuja el arco estrecho en la sección central
    draw_narrow(image, direction)

    return image


def randomize_and_draw_layout_for_open(direction: Direction) -> np.ndarray:
    """
    Genera el campo de juego para las rondas de desafío abierto.

    Devuelve un arreglo tridimensional de NumPy (matriz) que representa el
    campo de juego donde cada píxel está representado por tres números que
    corresponden al color BGR.
    """

    return draw_layout(direction, fixed=False)


def randomize_and_draw_layout_fixed(direction: Direction) -> np.ndarray:
    """
    Genera el campo de juego para las rondas de desafío abierto con el centro
    fijo.

    Devuelve un arreglo tridimensional de NumPy (matriz) que representa el
    campo de juego donde cada píxel está representado por tres números que
    corresponden al color BGR.
    """

    return draw_layout(direction, fixed=True)


def randomize_and_draw_layout_for_obstacle(direction: Direction) -> np.ndarray:
    """
    Genera el campo de juego para las rondas de desafío de obstáculos.

    Devuelve un arreglo tridimensional de NumPy (matriz) que representa el
    campo de juego donde cada píxel está representado por tres números que
    corresponden al color BGR.
    """

    # El conjunto de intersecciones que estarán frente al vehículo
    # en la zona de inicio para la dirección de conducción dada.
    forbidden_intersections = forbidden_intersections_in_start_zone[direction]

    # Busca los conjuntos de obstáculos que satisfacen las condiciones:
    #
    # - la diferencia entre el número de obstáculos verdes y rojos no es
    # mayor que uno
    #
    # - el número total de obstáculos es al menos 5
    #
    # - hay al menos una zona de inicio válida para la combinación
    # de obstáculos dada
    satisfied = False

    while not satisfied:
        # Elige el índice del conjunto de obstáculos obligatorio.
        mandatory_set_color = choice([Color.GREEN, Color.RED])
        mandatory_obstacles_set = mandatory_obstacles_sets[mandatory_set_color]

        # Elige el índice del conjunto de obstáculos requerido.
        required_obstacles_set = choice(required_obstacles_sets)

        # Elige el índice del conjunto de obstáculos para una de las dos
        # secciones restantes.
        os1 = mandatory_obstacles_set

        while os1 == required_obstacles_set or os1 == mandatory_obstacles_set:
            os1 = randint(0, len(obstacles_sets) - 1)

        # Elige el índice del conjunto de obstáculos para la última sección
        # restante.
        os2 = mandatory_obstacles_set

        while os2 == required_obstacles_set or  \
            os2 == mandatory_obstacles_set or \
                os2 == os1:
            os2 = randint(0, len(obstacles_sets) - 1)

        chosen_obstacles_sets_indices = [
            mandatory_obstacles_set,
            required_obstacles_set,
            os1,
            os2
            ]

        # Calcula el número de obstáculos, el número de obstáculos verdes y
        # rojos y las zonas de inicio prohibidas para los conjuntos de
        # obstáculos elegidos.
        forbidden_start_zones = {}
        obstacles_set_conflicting_with_parking_section = set()
        obstacles_amount = 0
        green_amount = 0
        red_amount = 0

        for obstacles_set_index in chosen_obstacles_sets_indices:
            one_obstacles_set = obstacles_sets[obstacles_set_index]

            obstacles_amount = obstacles_amount + len(one_obstacles_set)

            forbidden_start_zones[obstacles_set_index] = set()

            for one_obstacle in one_obstacles_set:
                if one_obstacle.is_green():
                    green_amount = green_amount + 1

                elif one_obstacle.is_red():
                    red_amount = red_amount + 1

                else:
                    raise ValueError("Unknown obstacle color")

                # Comprueba si el obstáculo actual estaría frente al
                # vehículo para cada posible zona de inicio en esta sección
                for zone in forbidden_intersections:
                    if one_obstacle.position in forbidden_intersections[zone]:
                        forbidden_start_zones[obstacles_set_index].add(zone)

                # Comprueba si la posición del obstáculo actual es adecuada
                # para la sección donde se encuentra el estacionamiento.
                for intersection in forbidden_intersections_in_parking_section:
                    if one_obstacle.position == intersection:
                        obstacles_set_conflicting_with_parking_section.add(
                            obstacles_set_index
                            )

        # Elimina los conjuntos de obstáculos donde ambas zonas de inicio
        # posibles están prohibidas,
        # manteniendo solo los conjuntos que tienen al menos una zona de
        # inicio válida.
        for obstacles_set_index in forbidden_start_zones:
            if len(forbidden_start_zones[obstacles_set_index]) == 2:
                del forbidden_start_zones[obstacles_set_index]

        # Obtén todos los conjuntos de obstáculos que son adecuados para la
        # sección de estacionamiento.
        obstacles_set_suitable_for_parking_section = set(
            chosen_obstacles_sets_indices
            ) - \
            obstacles_set_conflicting_with_parking_section

        # Detiene la búsqueda de los conjuntos de obstáculos si se satisfacen
        # las condiciones:
        #
        # - la diferencia entre el número de obstáculos verdes y rojos no es
        # mayor que uno
        #
        # - el número total de obstáculos es al menos 5
        #
        # - hay al menos una zona de inicio válida para la combinación de
        # obstáculos dada
        #
        # - hay al menos un conjunto de obstáculos que es adecuado para la
        # sección de estacionamiento
        satisfied = (abs(green_amount - red_amount) <= 1) and \
            (obstacles_amount > 4) and \
            (len(forbidden_start_zones) > 0) and \
            (len(obstacles_set_suitable_for_parking_section) > 0)

    # No se puede usar list(Section) porque los elementos de Section son
    # funciones.
    sections = [Section.NORTH, Section.WEST, Section.SOUTH, Section.EAST]

    # Asigna aleatoriamente cada conjunto de obstáculos a una sección única
    # del campo de juego
    shuffled_sections = sample(sections, 4)
    sections_for_obstacles_sets = {}
    for obstacles_set_index in chosen_obstacles_sets_indices:
        sections_for_obstacles_sets[obstacles_set_index] =  \
            shuffled_sections.pop()

    # Elige uno de los conjuntos de obstáculos que tiene al menos una zona de
    # inicio válida.
    obstacles_set_in_start_section = choice(list(forbidden_start_zones.keys()))

    # Elige la sección donde se encuentra el conjunto de obstáculos elegido.
    start_section = sections_for_obstacles_sets[obstacles_set_in_start_section]

    # Elige uno de los conjuntos de obstáculos que es adecuado para la sección
    # de estacionamiento.
    obstacles_set_in_parking_section = choice(
        list(obstacles_set_suitable_for_parking_section)
        )

    # Elige la sección donde se encuentra el conjunto de obstáculos elegido.
    parking_section = sections_for_obstacles_sets[
        obstacles_set_in_parking_section
        ]

    # Elige una de las zonas de inicio válidas para el conjunto de obstáculos
    # elegido.
    start_zone = choice(
        list(set([StartZone.Z3, StartZone.Z4]) -
             forbidden_start_zones[obstacles_set_in_start_section]
             ))

    scheme = {
        'start_section': start_section,
        'start_zone': start_zone,
        'obstacles': sections_for_obstacles_sets,
        'parking_section': parking_section
    }
    image = draw_scheme_for_final(scheme)

    # Dibuja las paredes interiores
    InnerWall().draw(image)

    # Dibuja el arco estrecho en la sección central
    draw_narrow(image, direction)

    return image


# HTTP Content related


def generate_image(img):
    """
    Codifica la imagen desde una matriz tridimensional de NumPy al formato PNG
    y la devuelve como respuesta HTTP.
    """

    res, im_png = cv2.imencode('.png', img)
    image = im_png.tobytes()
    response = make_response(image)
    response.headers.set('Content-Type', 'image/png')
    response.headers.set('Cache-Control', 'no-store')
    return response


# HTTP endpoints


@app.route('/')
def index():
    return render_template('index.html')


def random_direction():
    return random.choice([Direction.CW, Direction.CCW])


@app.route('/qualification/random')
def generate_qualification_random():
    direction = random_direction()
    layout = randomize_and_draw_layout_for_open(direction)
    response = generate_image(layout)
    return response


"""
# @app.route('/qualification/cw')
# def generate_qualification_cw():
#     layout = randomize_and_draw_layout_for_open(Direction.CW)
#     response = generate_image(layout)
#     return response


# @app.route('/qualification/ccw')
# def generate_qualification_ccw():
#     layout = randomize_and_draw_layout_for_open(Direction.CCW)
#     response = generate_image(layout)
#     return response
"""


@app.route('/qualification-fixed/random')
def generate_fixed_qualification_random():
    direction = random_direction()
    layout = randomize_and_draw_layout_fixed(direction)
    response = generate_image(layout)
    return response


"""
# @app.route('/qualification-fixed/cw')
# def generate_fixed_qualification_cw():
#     layout = randomize_and_draw_layout_fixed(Direction.CW)
#     response = generate_image(layout)
#     return response


# @app.route('/qualification-fixed/ccw')
# def generate_fixed_qualification_ccw():
#     layout = randomize_and_draw_layout_fixed(Direction.CCW)
#     response = generate_image(layout)
#     return response
"""


@app.route('/final/random')
def generate_final_random():
    direction = random_direction()
    layout = randomize_and_draw_layout_for_obstacle(direction)
    response = generate_image(layout)
    return response


"""
# @app.route('/final/cw')
# def generate_final_cw():
#     layout = randomize_and_draw_layout_for_obstacle(Direction.CW)
#     response = generate_image(layout)
#     return response


# @app.route('/final/ccw')
# def generate_final_ccw():
#     layout = randomize_and_draw_layout_for_obstacle(Direction.CCW)
#     response = generate_image(layout)
#     return response
"""

if __name__ == '__main__':
    app.run()
