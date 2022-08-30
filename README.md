# S-FaceNet

<p align="center"> 
  <img src="https://user-images.githubusercontent.com/40158342/183629498-886ecb9c-75c4-474b-a432-8d525a9f9691.png" height="300"> <br/>
  圖一、流程圖
</p>

#### Step 1. 下載資料集(放到datasets資料夾)
  - 訓練集：CASIA_WebFace、VGG-Face2
  - 測試集：LFW、CFP-FP、AgeDB

#### Step 2. 整理訓練集(訓練集優化)
  <p align="center">
    <img src="https://user-images.githubusercontent.com/40158342/183627978-3b160c35-2710-4628-a4f8-6a707782bd71.png" height="400"> <br/>
    圖二、優化資料集流程圖
  </p>
  
  - align資料夾 的 face_align.py：將不合適的人臉相片移除(流程寫在程式碼註解)
  - align資料夾 的 test.py：可用來確定單張照片的 bounding-box 及 人臉五官 的位置
  - datasets資料夾 的 annotation_generation.py：生成訓練集的標籤檔案(lable)
  - datasets資料夾 的 reconstruct_agedb.py：原AgeDB資料集人物相片並未依類別放進對應人物資料集中
  - datasets資料夾 的 build_agedb_pairs.py、build_cfpfp_pairs.py：將測試資料集的比對檔案(protocol)整理成輸入格式
  - 註：測試資料集也需要進行整理，為了符合真實人臉辨識系統的輸入格式。

#### Step 3. 訓練ing...
  - root路徑 的 Train.py：可以修改parser的參數(learning rate, epoch, feature_dim, traing_dataset...)
  - root路徑 的 Evaluation.py：__main__可以單獨驗證模型對某一測試資料集的檢測效果

#### Step 4. 畫訓練成果圖
  - root路徑 的 drawplot.py：可以畫出每隔 3000 iterations 的正確率

#### Step 5. 在PC架設簡易人臉辨識系統
  - root路徑 的 Take_picture.py：透過 WebCam 拍攝畫面中人臉並依據 parser 的 `--name`  給名
  - root路徑 的 Take_ID.py：讀取相片並依據 parser 的 `--name`  給名
  - root路徑 的 facebank.py：製作人臉資料庫，儲存已知人臉的特徵向量檔案
  - root路徑 的 Cam_demo.py：藉由 WebCam 拍攝影像 並 進行辨識
  - root路徑 的 Video_demo.py：輸入影片 並 進行人臉辨識
<p align="center">
  <img src="https://user-images.githubusercontent.com/40158342/183633830-d1e9bd77-d186-45f1-8fd5-818ad367c266.png" height="200">
  <img src="https://user-images.githubusercontent.com/40158342/183633819-6e438757-a043-43e5-b259-54605e11237f.png" height="200"> <br/>
  圖三、結果圖
</p>

### Reference
- Andrew G. Howard et al., "Mobilenets: Efficient convolutional neural networks for mobile vision applications," arXiv preprint arXiv:1704.04861, 2017.
- Sheng Chen et al., "Mobilefacenets: Efficient cnns for accurate real-time face verification on mobile devices," Chinese Conference on Biometric Recognition, 2018.
- xuexingyu24's MobileFaceNet Tutorial Pytorch Repository, https://github.com/xuexingyu24/MobileFaceNet_Tutorial_Pytorch
