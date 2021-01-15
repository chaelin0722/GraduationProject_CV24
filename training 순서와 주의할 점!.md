### training 순서와 주의할 점!

** 파일명에 공백, 괄호와 같은 기호 (_은 제외) 는 없어야 한다. 안그러면 python 인코딩문제가 발생

0.사진이 너무 크면 안됩니다. 200KB보다 적어야 하며, 720x1280 크기보다는 작아야 한다.
   이미지가 너 무크면 학습 시간이 길어집니다. repository에 있는 resizer.py를 사용하면 크기를 재조정할 수 있습니다.

1. sizeChecker.py를 실행해서 박스를친 부분이 학습에 사용될 데이터로 적절한지 체크

C:\tensorflow1\models\research\object_detection> python sizeChecker.py --move 

2. 먼저, 모든 train과 test 폴더에 있는 이미지들이 포함된 xml 데이터를 csv 파일로 바꾼다.
    \object_detection 폴더에서, anaconda 명령창에서 밑의 명령어를 실행하세요 :

 C:\tensorflow1\models\research\object_detection> python xml_to_csv.py 

   (\object_detection\images 폴더에 train_labels.csv 과 test_labels.csv 가 생성될 것)

3. generate_tfrecord.py 수정
그 후에, generate_tfrecord.py를 텍스트 편집기로 열어봅니다. 
31번째 줄에 각 사물에 맞는 ID 번호가 적혀진 라벨 맵을 바꿔야 합니다. 
단계 5b에 있는 labelmap.pbtxt 파일과 같은 갯수여야 합니다.

4. labelmap.pbtxt 수정

5.  python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
=> train.record 생성

5. python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record 
=> test.record 생성 