# Recycle-Auto-Machine

## 분리수거 로봇
<img src="https://github.com/ikefreet/Projects-images/blob/main/recycle%20robot/%EB%B6%84%EB%A6%AC%EC%88%98%EA%B1%B0%20%EB%A1%9C%EB%B4%87.PNG">

## 분리수거 로봇 구조도
<img src="https://github.com/ikefreet/Projects-images/blob/main/recycle%20robot/%EB%B6%84%EB%A6%AC%EC%88%98%EA%B1%B0%20%EB%A1%9C%EB%B4%87%20%EA%B5%AC%EC%A1%B0%EB%8F%84.PNG">

## 개요
젯슨 나노 보드에서 Yolov5 nano로 훈련시킨 모델로 캔, 페트병, 유리병 3종류의 쓰레기를 인식하게 하고 인식한 쓰레기를 카메라 프레임 기준 좌, 우, 중앙에 있는 것을 연산하여 차량을 움직이게 한다.

차량이 자동으로 쓰레기가 중앙에 오고 일정 거리 가까워지면 인식한 쓰레기의 종류에 쓰레기를 페트병이면 오른쪽에, 유리병이면 중앙에, 캔이면 왼쪽에 담기도록 쓰레받이의 서보 모터가 움직이고 그대로 로봇팔이 쓸어담는다.

## 결과 영상 링크 : https://youtu.be/rBhlxE6guC8
