# YouTube 조회수 예측: 감성적 피처 기반 ML 모델

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 프로젝트 개요

### 연구 배경 및 계기

최근 유튜브 플레이리스트 콘텐츠는 브랜드의 이미지를 높이거나 제품을 홍보하기 위해 **브랜디드 콘텐츠**로 활용되고 있습니다. 이는 단순한 제품 홍보를 넘어 소비자들의 반응과 공감을 이끌어내며 **브랜드와 소비자 간의 정서적 연결**을 형성하고 있습니다.

하지만 기존 연구는 다음과 같은 한계가 있었습니다:

- 먹방, 뷰티 등 특정 장르 콘텐츠 예측 모델은 존재하나, **플레이리스트 콘텐츠 관련 예측 모델은 부족**
- 다른 장르 연구들은 매력도와 같은 이미지의 **'한 요소'만 고려**하거나, 템포와 같은 오디오의 **'정량적' 요소만 고려**

→ **본 연구는 플레이리스트 콘텐츠 특성을 반영하여 감정적 요소를 중점으로 섬네일 이미지, 제목, 오디오를 통합적으로 고려한 유튜브 플레이리스트 조회수 예측 모델을 개발**했습니다.

### 프로젝트 목표

YouTube 플레이리스트 영상의 **감성적 피처(썸네일, 오디오, 제목)**를 통합 분석하여 조회수를 예측하는 머신러닝 모델을 구축합니다.

### 🎯 주요 특징

- **썸네일 분석**: 텍스트 비율, 상위 색상 10개, 밝기 대비, 질감, 총 색상 수, 얼굴 탐지, 객체 탐지, 색상 테마 일관성
- **오디오 분석**: 7가지 감정 분류 (`happy`, `sad`, `angry`, `fear`, `surprise`, `disgust`, `neutral`), 음악적 특성 (BPM, 피치, 에너지, 발화 속도 등)
- **제목 분석**: 이모지 여부, 특수문자, 문자 길이, 해시태그 개수
- **시간대 분석**: 업로드 시간대 원핫인코딩, 시간대-콘텐츠 매칭

### 📊 연구 성과

- **학회 발표**: 2025년 7월 4일 한국디지털콘텐츠학회 하계종합학술대회
- **논문 저자**: 변우중, 김홍인, 이진범
- **Best Model**: XGBoost (RMSE 1.9170)
- **평가 지표**: RMSE (Root Mean Squared Error)
- **Target Variable**: 로그 변환된 조회수 (`log_views`)
- **주요 발견**: 감성적 피처가 조회수 예측에 유의미한 영향

---

## 📁 프로젝트 구조

```
youtube-playlist-MLproject/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/                           # 데이터 디렉토리 (Git 제외)
│   ├── raw/                        # 원본 데이터
│   ├── processed/                  # 처리된 피처
│   └── models/                     # 저장된 모델
│
├── src/                            # 소스 코드
│   ├── features/                   # 피처 추출 모듈
│   │   ├── __init__.py
│   │   ├── thumbnail_features.py  # 썸네일 피처 (통합)
│   │   ├── face_detection.py      # 얼굴 탐지
│   │   ├── audio_quantitative.py  # 오디오 정량적 피처
│   │   ├── audio_qualitative.py   # 오디오 감정 분석
│   │   └── text_features.py       # 제목 피처
│   │
│   ├── preprocessing/              # 데이터 전처리
│   │   ├── __init__.py
│   │   ├── data_merger.py         # 피처 병합
│   │   └── feature_engineering.py # 피처 엔지니어링
│   │
│   ├── modeling/                   # 모델링
│   │   └── __init__.py
│   │
│   └── utils/                      # 유틸리티
│       ├── __init__.py
│       └── helpers.py              # 공통 함수
│
├── notebooks/                      # Jupyter 노트북
│   └── model_analysis.ipynb       # 모델 분석 및 시각화
│
└── scripts/                        # 실행 스크립트
    ├── extract_features.py        # 피처 추출 실행
    └── train_model.py             # 모델 학습 실행
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/byeonwoojung/youtube-playlist-MLproject.git
cd youtube-playlist-MLproject

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

데이터는 Git에서 제외되어 있습니다. 다음 구조로 데이터를 준비하세요:

```
data/
├── raw/
│   ├── youtubeInfo/
│   │   └── allYoutubeInfo_themeFiltered.csv
│   ├── thumbnails/
│   │   └── (썸네일 이미지 파일들)
│   ├── audio/
│   │   └── (오디오 파일들)
│   └── titles/
│       └── titles_final.csv
```

### 3. 피처 추출

```bash
# 모든 피처 한 번에 추출
python scripts/extract_features.py
```

또는 개별 모듈 실행:

```python
from src.features.thumbnail_features import ThumbnailFeatureExtractor
from src.preprocessing.data_merger import merge_all_features
from src.preprocessing.feature_engineering import engineer_features

# 1. 썸네일 피처 추출
extractor = ThumbnailFeatureExtractor(google_credentials_path="your_credentials.json")
extractor.extract_all_features(
    image_folder="data/raw/thumbnails",
    output_dir="data/processed/thumbnails"
)

# 2. 데이터 병합
df_merged = merge_all_features(
    base_info_path="data/raw/youtubeInfo/allYoutubeInfo_themeFiltered.csv",
    thumbnail_dir="data/processed/thumbnails",
    audio_dir="data/raw/audio",
    titles_path="data/raw/titles/titles_final.csv",
    output_path="data/processed/data_merged.csv"
)

# 3. 피처 엔지니어링
df_final = engineer_features(
    input_csv="data/processed/data_merged.csv",
    output_csv="data/processed/final_data.csv"
)
```

### 4. 모델 학습 및 분석

```bash
# Jupyter Notebook으로 분석
jupyter notebook notebooks/model_analysis.ipynb
```

---

## 📊 주요 피처

> **Note**: 본 섹션은 **실제 모델링에 사용된 피처**만 포함합니다.  
> 추출되었으나 모델링에서 제외된 피처들: `subscriber_count`, `days_before_reference_ceiled`, `colorsDaily_matchScore`, `colorsSensibility_matchScore` (통합된 `colorsTheme_matchScore` 사용), `brightness_weightedStd`, `texture_sharpness_scaled`, `colorsCluster_0~44` (상위 10개만 사용)### 1. 썸네일 피처 (Thumbnail Features)

| 피처                                             | 설명                         | 추출 방법          | 모델 사용 |
| ------------------------------------------------ | ---------------------------- | ------------------ | --------- |
| `text_ratio`                                   | 썸네일 내 텍스트 면적 비율   | Google Vision OCR  | ✅        |
| `colorRank_1~10`                               | 상위 10개 색상 클러스터 비율 | K-Means 클러스터링 | ✅        |
| `total_colors`                                 | 사용된 색상 클러스터 개수    | 색상 다양성        | ✅        |
| `brightness_contrast`                          | 밝기 대비 (Sigmoid 변환)     | Grayscale 분석     | ✅        |
| `texture_sharpness`                            | 질감 및 선명도               | Laplacian 변환     | ✅        |
| `person`, `animal`, `anime`, `landscape` | 객체 탐지                    | GPT-4o Vision API  | ✅        |
| `total_faces`, `frontal_faces_8_percent`     | 얼굴 탐지 및 정면 얼굴       | YuNet 모델         | ✅        |
| `colorsTheme_matchScore`                       | 색상 감성•일상 테마 일치도 | LAB 거리 기반 매칭 | ✅        |

### 2. 오디오 피처 (Audio Features)

| 피처                                       | 설명           | 추출 방법         | 모델 사용 |
| ------------------------------------------ | -------------- | ----------------- | --------- |
| **감정 피처 (Audio Emotional)**      |                |                   |           |
| `happy`                                  | 기쁨 감정 확률 | Wav2Vec2 모델     | ✅        |
| `sad`                                    | 슬픔 감정 확률 | Wav2Vec2 모델     | ✅        |
| `angry`                                  | 분노 감정 확률 | Wav2Vec2 모델     | ✅        |
| `fear`                                   | 공포 감정 확률 | Wav2Vec2 모델     | ✅        |
| `surprise`                               | 놀람 감정 확률 | Wav2Vec2 모델     | ✅        |
| `disgust`                                | 혐오 감정 확률 | Wav2Vec2 모델     | ✅        |
| `neutral`                                | 중립 감정 확률 | Wav2Vec2 모델     | ✅        |
| **음악적 특성 (Audio Quantitative)** |                |                   |           |
| `pitch_mean`                             | 평균 피치      | librosa yin       | ✅        |
| `energy_mean`                            | 평균 에너지    | RMS               | ✅        |
| `centroid_mean`                          | 스펙트럼 중심  | Spectral Centroid | ✅        |
| `bpm`                                    | 템포 (BPM)     | Beat Tracking     | ✅        |
| `speech_rate`                            | 발화 속도      | Onset Detection   | ✅        |
| `initial_silence`                        | 초기 침묵 시간 | RMS 임계값        | ✅        |

### 3. 제목 피처 (Title Features)

| 피처                         | 설명               | 모델 사용 |
| ---------------------------- | ------------------ | --------- |
| `has_emoji`                | 이모지 포함 여부   | ✅        |
| `has_question_exclamation` | 물음표/느낌표 여부 | ✅        |
| `char_length`              | 공백 제외 문자 수  | ✅        |
| `hashtag_count`            | 해시태그 개수      | ✅        |

### 4. 메타 피처 (Engineered Features)

| 피처                                                                                 | 설명                         | 모델 사용 |
| ------------------------------------------------------------------------------------ | ---------------------------- | --------- |
| `time_midnight`, `time_morning`, `time_noon`, `time_evening`, `time_night` | 업로드 시간대                | ✅        |
| `time_match_content`                                                               | 시간대-콘텐츠 타입 매칭 여부 | ✅        |
| `text_char_combo`                                                                  | 썸네일 텍스트 × 제목 길이   | ✅        |
| `object_complexity`                                                                | 객체 요소 복잡도             | ✅        |

---

## 🧪 모델 성능

### 평가 지표

본 연구에서는 **RMSE (Root Mean Squared Error)**를 주요 평가 지표로 사용했습니다.

- **Target Variable**: 로그 변환된 조회수 (`log_views`)
- 로그 변환을 통해 조회수의 왜도를 정규화하고 이상치 영향 완화

### 실험 결과

**데이터셋 구성**

- Total: 6,826건
- Train: 5,460건
- Test: 1,366건

**모델 평가 결과 (RMSE)**

| 모델              | RMSE             | 비고                    |
| ----------------- | ---------------- | ----------------------- |
| Random Forest     | 1.9269           | -                       |
| **XGBoost** | **1.9170** | **Best Model** ✅ |
| LightGBM          | 1.9416           | -                       |

→ **평균제곱근 오차(RMSE)의 값이 가장 낮은 XGBoost 선택**

### 주요 Feature Importance (Top 10)

1. `subscriber_count` - 구독자 수
2. `audio_emotional` -`happy `, `sad` 등 감정 확률
3. `text_ratio` - 썸네일 텍스트 비율
4. `colorsCluster_*` - 색상 클러스터 분포
5. `frontal_faces_8_percent` - 정면 얼굴 수
6. `bpm` - 음악 템포
7. `brightness_std` - 밝기 대비
8. `time_match_content` - 시간대 매칭
9. `object_complexity` - 객체 복잡도
10. `hashtag_count` - 해시태그 개수

### 연구 결론 및 제안

본 연구를 통해 다음과 같은 결론을 도출했습니다:

#### 메타적 요소

- **유튜브 특성상 해시태그 수는 알고리즘과 직접 연결됨**: 크리나, 해시태그 수가 과도하게 많을수록 오히려 스팸으로 인식될 가능성이 있으므로 역효과 추의

#### 썸네일

- **텍스트 비율이 높을 시 과도한 정보 전달로 인해 역효과가 능성이 있음**: 명확 대비가 낮은 (전체적으로 밝거나 어두운) 섬네일은 섬네일은 실내에 적절히 부정적 영향성
- **업로드 시간대가 밤(21~24시)일 때 상대적으로 낮은 조회수 경향성**

#### 제목

- **이모티콘 비율이 과도하게 높을 시 심리도를 하락시키고, 상반되게 낮을 수 있음**: 감성·일상 테마의 플레이리스트 콘텐츠에서는 **자극적이고 관심/흥미를 유도하는 표현**이 많을수록 조회수 증가하는 경향이 있음

#### 오디오

- **청공 1분 오디오의 스펙트럼 평균 조파수가 낮을수록 부드럽고 감성적인 분위기가 낮춰 수 있음**: **오디오 감정이 기쁨(happy), 슬픔(sad)일 때** 감성·일상 테마 콘텐츠의 정서와 조회수가 높은 경향 있음

---

## 🛠 기술 스택

### 핵심 라이브러리

- **데이터 처리**: pandas, numpy
- **이미지 처리**: OpenCV, Pillow
- **머신러닝**: scikit-learn, XGBoost, LightGBM, Optuna (트리 기반 모델)
- **딥러닝**: TensorFlow, PyTorch, transformers
- **오디오 처리**: librosa, yt-dlp
- **시각화**: matplotlib, seaborn
- **OCR**: Google Cloud Vision API

### 연구 방법론

- **모델링 기법**: 트리 기반 머신러닝 모델 (Random Forest, XGBoost, LightGBM)
- **하이퍼파라미터 튜닝**: Optuna
- **피처 엔지니어링**: 감정 분석, 이미지 처리, 오디오 신호 처리

### 하드웨어 가속

- **GPU 지원**: CUDA (PyTorch, TensorFlow)
- **멀티프로세싱**: concurrent.futures

---

## 📝 사용 예시

### 썸네일 피처 추출

```python
from src.features.thumbnail_features import ThumbnailTextExtractor, ThumbnailColorExtractor

# 텍스트 비율 추출
text_extractor = ThumbnailTextExtractor(
    credentials_path="google_credentials.json",
    max_workers=4
)
text_df = text_extractor.extract_batch(
    image_folder="data/thumbnails",
    output_csv="data/thumbnails_text.csv"
)

# 색상 클러스터 추출
color_extractor = ThumbnailColorExtractor(n_clusters=45)
color_df = color_extractor.extract_batch(
    image_folder="data/thumbnails",
    output_csv="data/thumbnails_colors.csv"
)
```

### 얼굴 탐지

```python
from src.features.face_detection import GPUFaceAnalyzer

analyzer = GPUFaceAnalyzer(model_path='face_detection_yunet_2023mar.onnx')
results = analyzer.process_batch_images(
    image_paths=["thumb1.jpg", "thumb2.jpg"],
    output_dir="results",
    save_visualizations=True
)
```

### 데이터 병합 및 피처 엔지니어링

```python
from src.preprocessing.data_merger import merge_all_features
from src.preprocessing.feature_engineering import engineer_features

# 병합
df_merged = merge_all_features(
    base_info_path="data/youtube_info.csv",
    thumbnail_dir="data/thumbnails",
    audio_dir="data/audio",
    titles_path="data/titles.csv",
    output_path="data/merged.csv"
)

# 피처 엔지니어링
df_final = engineer_features(
    input_csv="data/merged.csv",
    output_csv="data/final_data.csv"
)
```

---

## 🔍 연구 배경

### 동기

YouTube 영상의 조회수는 다양한 요인에 영향을 받습니다. 기존 연구들은 주로 메타데이터(제목, 태그, 설명)에 집중했지만, 본 연구는 **감성적 요소**에 주목했습니다:

- 썸네일의 시각적 매력도
- 오디오의 감정적 반응
- 업로드 시간대와 콘텐츠 타입의 조화

### 연구 방법론

1. **데이터 수집**: YouTube API를 통한 영상 메타데이터 수집
2. **피처 추출**:
   - 썸네일: OCR, 색상 분석, 객체/얼굴 탐지
   - 오디오: 감정 분류, 음악적 특성
   - 제목: 텍스트 분석
3. **피처 엔지니어링**: 시간대 매칭, 복합 피처 생성
4. **모델링**: 트리 기반 머신러닝 모델 + Optuna 하이퍼파라미터 튜닝

---

## 📚 참고 자료

### 학회 논문

- **제목**: "섬네일·제목·오디오 기반 통합적 유튜브 플레이리스트 조회수 예측"
- **학회**: 한국디지털콘텐츠학회 하계종합학술대회
- **발표일**: 2025년 7월 4일
- **저자**: 변우중, 김홍인, 이진범

---

## 🤝 기여

이 프로젝트는 학술 연구 목적으로 개발되었습니다. 개선 제안이나 버그 리포트는 환영합니다!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 👤 저자

**변우중, 김홍인, 이진범**

- GitHub: [@byeonwoojung](https://github.com/byeonwoojung), [@hongin12](https://github.com/hongin12)
- Repository: [youtube-playlist-MLproject](https://github.com/byeonwoojung/youtube-playlist-MLproject)

---

## 🙏 감사의 글

본 연구는 **SK 플래닛 T아카데미 ASAC 빅데이터 분석가** 과정을 통해 수행되었습니다.

- SK 플래닛 T아카데미 ASAC 프로그램
- 한국디지털콘텐츠학회
- 모든 오픈소스 기여자들

---

## 📮 문의

질문이나 제안 사항이 있으시면 GitHub Issues를 통해 연락해주세요.

**Happy Predicting! 📊🎬**
