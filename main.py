from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import time
import cv2
import json
from sse_starlette import EventSourceResponse

VIDEO_PATH = "videos/walking.mp4"
CONFIG_PATH = "yolov4-tiny.cfg"
WEIGHTS_PATH = "yolov4-tiny.weights"

app = FastAPI()

# Configurar permissões CORS
app.add_middleware(
  CORSMiddleware,
  allow_origins=['*'],
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*'],
)

# Simulação de uma função de detecção de pessoas
def detect_person():
    # CORES DAS CLASSES 
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

    # CARREGA AS CLASSES
    class_names = []
    with open("coco.names", "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]

    # CAPTURA DO VIDEO
    cap = cv2.VideoCapture(VIDEO_PATH)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)
    frames = 0
    # CARREGANDO OS PESOS DA REDE NEURAL
    #net = cv2.dnn.readNet("weights/yolov4.weights", "cfg/yolov4.cfg")
    net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)

    # SETANDO OS PARAMETROS DA REDE NEURAL
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255)

    # LENDO OS FRAMES DO VIDEO
    while True:
        
        # CAPTURA DO FRAME
        _, frame = cap.read()
        frames += 1
        
        # COMEÇO DA CONTAGEM DOS MS
        start = time.time()
        
        # DETECCAO
        classes, scores, boxes = model.detect(frame, 0.1, 0.2)
        
        # FIM DA CONTAGEM DOS MS
        end = time.time()
        
        ## INICIA O CONTADOR DE PESSOAS
        counter = 0
        detections = []
        
        # PERCORRER TODAS AS DETECCOES
        for (classid, score, box) in zip(classes, scores, boxes):
            
            # GERANDO UMA COR PARA A CLASSE
            color = COLORS[int(classid) % len(COLORS)]
            # --- contador ---
            if classid == 0:
                counter += 1
                detections.append(f"Person[{counter}]: {score*100:.2f}%")
            # -----------
            
            # PEGANDO O NOME DA CLASSE PELO ID E O SEU SCORE DE ACURACIA
            # label = f"{class_names[classid[0]]} : {score}"
            label = f"[{counter}] {class_names[classid]} : {score*100:.2f}%"
            
            # DESENHANDO A BOX DE DETECCAO
            cv2.rectangle(frame, box, color, 2)
            
            # ESCREVENDO O NOME DA CLASSE EM CIMA DA BOX DO OBJETO
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        # CALCULANDO O TEMPO QUE LEVOU PARA FAZER A DETECCAO
        fps = round((1.0/(end - start)),2)
        fps_label = f"Pessoas: {counter} FPS: {fps}"
        time_ms = end - start    
        # ESCREVENDO O FPS NA IMAGEM
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 5)
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
            
        # MOSTRANDO A IMAGEM
        cv2.imshow("detections", frame)
            
        # ESPERA DA RESPOSTA
        if cv2.waitKey(1) == 27:
            break
        
        yield json.dumps({
            'frame:': frames,
            'person': counter,
            'fps': fps,
            'detection_time': time_ms,
            'detections': detections
        })
        
    
    # LIBERACAO DA CAMERA E DESTROI TODAS AS JANELAS
    cap.release()
    cv2.destroyAllWindows()

@app.get("/events")
async def sse_endpoint():
    # Tentei com StreamingResponse, mas não estava dando certo
    return EventSourceResponse(detect_person(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
