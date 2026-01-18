
airline 단변량 시계열 데이터에서 월별 항공기 승객 수(Number of airline passengers)를 예측한다.

예제에서는 fh=[1,2,3]로 다음 3개월치를 forecast 한다.

Local 실행 방법

```
cd /root/Workspace/python_study/src/tmm
python3 -m pip install --break-system-packages -r requirements.txt
python3 scripts/run_ttm_airline_zeroshot.py
```

HTTP 파일 서버 실행 예시

```
cd /root/Workspace/python_study/src/tmm
python3 http_octet_server.py --directory ./ --port 8080
```

API 호출 방법
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
