# WRO 2024 Future Engineers Aplicación de aleatorización

La aplicación web simplifica el proceso de aleatorización preparando una imagen con diseños aleatorios para desafíos abiertos y de obstáculos:

- Para el desafío abierto
  - La configuración de las paredes interiores
  - La zona inicial

- Para el desafío de obstáculos
  - La posición de los obstáculos
  - la zona inicial
  - La sección del estacionamiento

Ejemplos de las imágenes:

| Desafío abierto | Desafío de obstáculos |
|:----:|:----:|
| ![image](https://github.com/user-attachments/assets/eab032fb-20b3-4eff-9d0a-32404de0ced8) | ![image](https://github.com/user-attachments/assets/937f0b5e-c089-4d7b-8c17-16c25cee9abc) |

## Ejecutar desde CLI

- `sudo apt-get update`
- `sudo apt-get install -y libgl1-mesa-glx`
- `pip install -r requirements.txt`
- `gunicorn -w 2 aplicacion:app`

## Ejecutar por Docker

- `docker build -t fe-randomization-app .`
- `docker run -ti --rm -e HOST=0.0.0.0 -e PORT=8000 -p 8000:8000 fe-randomization-app`
