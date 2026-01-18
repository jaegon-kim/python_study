TMM(Tiny Time Mixer)
---

airline 단변량 시계열 데이터에서 월별 항공기 승객 수(Number of airline passengers)를 예측한다.
예제에서는 fh=[1,2,3]로 다음 3개월치를 forecast 한다.


## 컨테이너 구동 방법
https://github.com/jaegon-kim/my-containers/blob/main/k8s-deployment-examples/ubuntu/ubuntu-dev-1.yaml
를 이용하여 모델을 서빙한다. 위 ubuntu-dev-1.yaml 은 8080 포트를 열고 있는 Service도 함께 생성한다.

만약 ubuntu-dev 이미지가 없는 상태라면, 
https://github.com/jaegon-kim/my-containers/blob/main/k8s-deployment-examples/ubuntu/Makefile
를 이용하여 make image, make push 해서 이미지를 만들어서 docker registry에 등록해야 한다.

컨테이너 파드를 생성한다.
```
kubectl apply -f ubuntu-dev-1.yaml
```

컨테이너 파드에 접속한다.
```
kubectl exec -it ubuntu-dev-1 -- /bin/bash
```

## Codex
만약 컨테이너 안에서 codex를 실행시키고 싶다면, 다음과 같이 수행할 수 있다.
```
# npm 전역 패키지 권한 설정 (권장)
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Codex CLI 설치
npm install -g @openai/codex

# 인증
codex
```


## Local 실행 방법

```
cd /root/Workspace/python_study/src/tmm
python3 -m pip install --break-system-packages -r requirements.txt
python3 scripts/run_ttm_airline_zeroshot.py
```

## 모델 서빙 HTTP 파일 서버 실행 예시

```
cd /root/Workspace/python_study/src/tmm
python3 http_octet_server.py --directory ./ --port 8080
```

## 모델 빌드

```
make triton
```
- venv를 다운로드 받고, Hugging Face에서 가중 치를 다운로드 받아 triton_model_repository.tar.gz 파일을 생성한다.

```
make clean
```
triton_model_repository.tar.gz 를 삭제한다.

```
make clean-all
```
triton_model_repository.tar.gz, venv, model 가중치를 모두 삭제한다.

## API 호출 방법
```
HOST=ttm-triton-predictor.default.211.226.3.148.nip.io
NODE_IP=211.226.3.148
INGRESS_PORT=30173
MODEL_NAME=ttm

curl -v -sS -X POST -H "Host: ${HOST}" \
  "http://${NODE_IP}:${INGRESS_PORT}/v2/models/${MODEL_NAME}/infer" \
  -H 'Content-Type: application/json' \
    -d '{
      "inputs": [
        {
          "name": "y",
          "shape": [12],
          "datatype": "FP32",
          "data": [112.0,118.0,132.0,129.0,121.0,135.0,148.0,148.0,136.0,119.0,104.0,118.0]
        },
        {
          "name": "fh",
          "shape": [3],
          "datatype": "INT64",
          "data": [1,2,3]
        }
      ],
      "outputs": [
        {"name": "y_pred"}
      ]
    }'
```
