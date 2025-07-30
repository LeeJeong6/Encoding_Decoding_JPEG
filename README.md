# JPEG Compression and Decompression (Python Implementation)

본 프로젝트는 JPEG 압축 방식을 파이썬으로 구현한 학습용 코드입니다.  
이미지를 입력받아 JPEG의 주요 단계를 거쳐 인코딩/디코딩을 수행하고,  
각 단계별 메모리 사용량 변화를 확인할 수 있습니다.

개인 유튜브로 구현 설명 영상을 올릴 계획입니다.
---

## JPEG 압축 과정

JPEG 압축은 크게 6단계로 이루어집니다.

1. **RGB → YCbCr 변환**
   - RGB 이미지를 사람의 시각 특성에 맞게 Y (밝기), Cb/Cr (색차) 성분으로 변환합니
   - 밝기에 민감하고 색차에는 둔감한 인간의 눈 특성을 활용하기 위함

2. **다운샘플링 (Downsampling)**
   - Cb, Cr 채널을 2x2 블록 단위로 줄여 해상도를 절반으로 축소합니다.  
   - 예: 512×512 → 256×256
   - 데이터 양을 크게 줄이는 핵심 단계

3. **DCT (이산 코사인 변환)**
   - 이미지를 8×8 블록으로 나눠 주파수 영역으로 변환
   - 저주파(좌상단)는 영상의 전반적인 윤곽, 고주파(우하단)는 세부 디테일 표현

4. **양자화 (Quantization)**
   - 사람 눈에 덜 중요한 고주파 계수를 더 거칠게 줄입니다.
   - 이 과정에서 많은 계수가 0이 되어 이후 압축 효율을 높임.


5. **RLE (Run-Length Encoding)**
   - 지그재그 스캔 후, 연속된 0의 개수를 기록하여 데이터 줄임.


6. **허프만 부호화 (Huffman Coding)**
   - 등장 빈도 기반의 가변 길이 이진 코드로 무손실 압축
   - 자주 등장하는 값은 짧은 비트코드를, 드문 값은 긴 비트코드를 할당

---

##  디코딩 과정

압축된 데이터를 다시 원본 이미지로 복원합니다.

1. 허프만 디코딩 → RLE 역변환
2. 지그재그 역스캔 → 8×8 블록 복원
3. 양자화 역처리 → 근사 DCT 계수 복구
4. IDCT (역이산 코사인 변환) → 픽셀 영역 복원
5. 업샘플링 (Cb, Cr) → YCbCr 재구성
6. YCbCr → RGB 변환

---


## 단계별 메모리 비교

1. 원본 이미지(512x512): 773.00 KB
2. YCbCr 분리: 3072.00 KB
3. 다운샘플링: 1536.00 KB
4. 양자화 후: 1536.00 KB
5. RLE 후: 143.22 KB
6. 허프만 후: 20.64 KB

## 메모리 차지 해석
- 원본은 RGB니까 int8로 저장되어있음(0~255니까)
- 이때 코드에서 YCbCr채널로 변경하면서 float32로 변경합니다. 여기서는 변경을 안해도되지만 나중에 결국 변경하기 때문에 먼저 했습니다
- float32로 저장하면서 4배커짐. 저장하는 자리수가 더 많아지기 때문
- 다운샘플링 후에 Cb,Cr픽셀수를 각각 1/4로 줄이니까 Y+1/4Cb+1/4Cr이라 절반으로 줄어듬
- 양자화는 그냥 나눗셈만 하는거임-> 이미지 자체로는 정보를 압축하는 과정이지만 메모리 차원에서 봤을 때는 변화 X
- RLE는 0이 길게 반복되는 부분을 하나로 줄이니가 데이터가 압축됨
- 허프만은 데이터 출현 빈도를 효율적으로 할당하니까 줄어든다
<img width="1354" height="598" alt="Image" src="https://github.com/user-attachments/assets/b9e10152-122c-47f9-bd19-0f7fe6c7deda" />

## 결과

**원본 이미지**
<img width="512" height="512" alt="Image" src="https://github.com/user-attachments/assets/f24cf9cc-f5ac-4fce-80de-6bf14e468835" />
---

**압축한 이미지**
![Image](https://github.com/user-attachments/assets/57bb4f8a-eeef-4d03-8258-b6e54000e195)

---

**비교사진**
<img width="1000" height="500" alt="Image" src="https://github.com/user-attachments/assets/ef99e1de-f203-46bb-bb7f-5d06eba4192a" />

## 실행 방법
```bash
pip install -r requirements.txt
```
```bash
python JPEG.py --이미지경로
```


