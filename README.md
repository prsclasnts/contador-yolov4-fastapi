# Contador de Pessoas com YOLO v4

Esta é uma aplicação web desenvolvida com FastAPI e o Darknet YOLO v4 para detecção e contagem de pessoas com envio de informações em tempo real.

## Pré-requisitos

- Python 3.x
- Biblioteca OpenCV
- Biblioteca Numpy
- Framework FastAPI
- Framework Starlette
- Vídeos ou câmera para teste
- Arquivos de configuração e pesos do YOLOv4 ou YOLOv4-tiny:
  - [yolov4-tiny.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg)
  - [yolov4-tiny.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights) 23.1MB
  - [yolov4.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg)
  - [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) 245MB

## Instruções

Clone este repositório em seu ambiente de desenvolvimento local:

```markdown
git clone https://github.com/prsclasnts/contador-yolov4-fastapi.git
```

Como forma de evitar conflitos com o ambiente de desenvolvimento local, estou trabalhando com um ambiente virtual de desenvolvimento. O código a seguir irá inicializar um ambiente virtual em Python:

```markdown
python -m venv .venv
```

Depois de criado, o ambiente virtual precisa ser ativado. Para ativá-lo no Windows use:

```markdown
./.venv/Scripts/activate
```

Com o ambiente virtual ativado, instale as dependencias manualmente ou utilize o arquivo requirements.txt com o comando:

```markdown
pip install -r requirements.txt
```

Faça o download dos arquivos de configuração e pesos da rede neural e os inclua na pasta do projeto.
No arquivo main.py altere o valor das constantes CONFIG_PATH e WEIGHTS_PATH para o caminho dos arquivos baixados.

São necessários arquivos de vídeo ou uma webcam para teste da detecção de pessoas. Podemos então trabalhar de duas formas:

### Com a Webcam

1. Certifique-se de ter uma Webcam instalada e que não esteja sendo utilizada em outra aplicação;
2. No arquivo main.py altere o valor da constante VIDEO_PATH para 0.

### Com arquivos de video

1. Inclua dentro da pasta do projeto um arquivo de vídeo preferencialmente contendo pessoas;
2. No arquivo main.py altere o valor da constante VIDEO_PATH para caminho do vídeo escolhido.

## Executando o código

Para testar o código precisamos executar tanto o código do cliente quanto o do servidor. Para executar o código do cliente utilizamos a extensão [Live Server](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) do VisualStudio Code com o arquivo index.html do projeto.

O código do servidor pode ser executado com o comando:

```markdown
uvicorn main:app --reload
```
