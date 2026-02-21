# pose-transformer-with-real-transoformer
이미지를 넣어서 pose에 따라 이미지를 변환시키는 건데 만들긴 한거 같은데 학습 자료가 별로 안 좋아서... 잘 안되는 거 같습니다. 일단 모델은 올려봅니다.
수정할 부분이나 그런거 있으면 좀 알려주세요.

제가 직접 만든 모델을 한번 구경해보고 싶어서요. 지금은 좀 엉망으로 나오는지라....

제 이메일등(ryujanghyun00@icloud.com)으로 자료 보내 주시면 학습 한번 해볼게요. 이미지 사이즈는 WxH가 144x256로 만들었습니댜.

사용 방법은 

1. python -m venv ./ 이후에 source bin/activate 작동.
2. pip install -r requirements.txt
3. pth.zip과 outdate.zip 그리고 fashion.zip을 압축 풀어 넣기. (학습시 fashion.zip or outdate.zip필요 실행만 해볼 요량이면 pth.zip만 압축 풀기)  ( 다운로드 경로 https://drive.google.com/drive/folders/1jNeGA9nbLsa37PXgRcy-lhROPh58zOnp?usp=drive_link  )
  (학습 순서
    1. fashion.zip파일을 푼 이후에
       python change_npnew.py
       그러면 outdate폴더에 npy파일이 생성됨.
    2. 이후
       python mytrain.py
       이러면 학습이 진행됨
       pth폴더에 torch모델이 저장됨.
  ) 
5. $ python mymain.py 실행
   (테스트시에 컴퓨터에 웹카메라등을 설치가 필요합니다. 웹 카메라는 90도 반시계 방향으로 돌린 이후 사용합니다.)
   (pth폴더의 파일의 이름을 확인한 이후 mymain.py에 파일 이름을 교체합니다.)
<img width="1920" height="1080" alt="스크린샷 2026-02-21 16-39-32" src="https://github.com/user-attachments/assets/1d7e2449-6312-4f4e-924a-5c75983dbfbe" />

참조 논문은
https://github.com/prasunroy/pose-transfer?tab=readme-ov-file 이걸 참조해서 사이에 transformer를 넣어 작성해 봤습니다. 더 잘되는 것 같긴 합니다.



2026/2/21
gan 모델을 추가하였습니다. 학습속도가 비약적으로 빨라 졌습니다.
그리고 loss 모델을 수정했습니다.

<img width="1920" height="1080" alt="스크린샷 2026-02-21 16-40-26" src="https://github.com/user-attachments/assets/3bfe0128-767a-41d9-a496-b2441dd98563" />
<img width="1920" height="1080" alt="스크린샷 2026-02-21 16-40-50" src="https://github.com/user-attachments/assets/5f39c71a-dbab-4f3c-8b35-d7391799dc88" />
<img width="1920" height="1080" alt="스크린샷 2026-02-21 16-49-34" src="https://github.com/user-attachments/assets/4f8f59c1-27ff-4c63-9bcb-40f5617826f6" />
<img width="1920" height="1080" alt="스크린샷 2026-02-21 17-31-10" src="https://github.com/user-attachments/assets/1584563a-f63f-4fda-8548-98a328966f8b" />
<img width="1920" height="1080" alt="스크린샷 2026-02-21 18-00-44" src="https://github.com/user-attachments/assets/a9996dec-a530-4ab0-89e0-3e103b374f51" />
<img width="1920" height="1080" alt="스크린샷 2026-02-21 20-05-05" src="https://github.com/user-attachments/assets/605c3b24-06cb-4e9c-84d2-fa53c3569a48" />
<img width="1920" height="1080" alt="스크린샷 2026-02-21 21-20-45" src="https://github.com/user-attachments/assets/98b1dc12-952d-4959-9753-0d8a2788def8" />
<img width="1920" height="1080" alt="스크린샷 2026-02-21 21-37-59" src="https://github.com/user-attachments/assets/b5816d08-d38b-4071-a299-29433d427cd7" />
<img width="1920" height="1080" alt="스크린샷 2026-02-21 22-26-36" src="https://github.com/user-attachments/assets/c108cc08-88ce-44e3-b6a3-f1d7e4510577" />


이건 30000step에서 확인 한 걸 돌려본겁니다. 2틀뒤 200000step에서 작동되는걸 올려보겠습니다.
![2026-02-21 22-32-51](https://github.com/user-attachments/assets/afa75695-19f1-484b-90e3-c8a5dd6ef3a8)








