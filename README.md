# OCR
projects about OCR application to get information cccd 

projects using yolov5 , pytesseract and flask

Step 1: use yolov5 to detect 4 corners of cccd

![fcc0c41d5a198447dd08](https://github.com/Duongvinh227/OCR/assets/96807833/e6ca66f0-fe1e-4d5d-8c09-3b4eb154491b)

Step 2: use function perspective_transform to rotate the photo to the correct position

![b135a968026cdc32857d](https://github.com/Duongvinh227/OCR/assets/96807833/209c4b4e-8031-4763-ae4f-afc5d34e133d)

Step 3 : detect locations containing information on cccd

Step 4: use pytesseract to get information about information storage locations and rewrite the information as json

Step 5 : finally displayed image and json information up

![d580e7dc3fd8e186b8c9](https://github.com/Duongvinh227/OCR/assets/96807833/568f2634-0a9c-4532-9a45-6b70e203758f)
