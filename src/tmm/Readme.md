
airline 단변량 시계열 데이터에서 월별 항공기 승객 수(Number of airline passengers)를 예측한다.

예제에서는 fh=[1,2,3]로 다음 3개월치를 forecast 한다.

Local 실행 방법

```
cd /root/Workspace/python_study/src/tmm
python3 -m pip install --break-system-packages -r requirements.txt
python3 scripts/run_ttm_airline_zeroshot.py
```