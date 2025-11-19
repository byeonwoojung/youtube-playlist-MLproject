"""
썸네일 피처 추출 모듈

YouTube 썸네일 이미지로부터 다양한 감성적 피처를 추출합니다:
- 텍스트 비율 (OCR 기반)
- 색상 클러스터 비율 (CSS3 45개 클러스터)
- 밝기 표준편차 (대비)
- 질감 및 선명도
- 객체 탐지 (사람, 동물, 애니메이션, 풍경)
- 색상 테마 매칭
- 얼굴 탐지 및 정면 얼굴 분석

참고: 오디오 감정 피처(Audio Emotional)는 audio_qualitative.py에서 처리됩니다.
      - happy, sad, angry, fear, surprise, disgust, neutral

⚠️ 모델링에서 제외된 피처:
   - colorsDaily_matchScore, colorsSensibility_matchScore 
     → 대신 이 둘의 최댓값인 colorsTheme_matchScore 사용
   - brightness_weightedStd (대신 brightness_weightedStd_scaledSigmoid 사용)
   - texture_sharpness_scaled (대신 원본 texture_sharpness 사용)
   - colorsCluster_0~44 전체 (모델링 시 상위 10개만 colorRank_1~10으로 변환)
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 상대 경로 import를 위한 설정
sys.path.append(str(Path(__file__).parent.parent))

# Google Vision API 설정
from google.cloud import vision
import re
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache

# 색상 관련 라이브러리
import webcolors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# 유틸리티 함수 import
try:
    from utils.helpers import setup_gpu, save_csv_safely
except ImportError:
    print("⚠ utils.helpers를 찾을 수 없습니다. 기본 함수를 사용합니다.")
    
    def setup_gpu():
        return {"device": "cpu"}
    
    def save_csv_safely(df, filepath, encoding="utf-8-sig"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False, encoding=encoding)
        return True


# ========================================
# 1. 텍스트 피처 추출 (OCR)
# ========================================

class ThumbnailTextExtractor:
    """
    썸네일 이미지에서 텍스트 비율 추출
    Google Cloud Vision API 사용
    """
    
    def __init__(self, credentials_path: str, max_workers: int = 4):
        """
        Args:
            credentials_path: Google Cloud 인증 JSON 파일 경로
            max_workers: 병렬 처리 워커 수
        """
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        self.client = vision.ImageAnnotatorClient()
        self.max_workers = max_workers
        self.lock = threading.Lock()
    
    @staticmethod
    def is_valid_text(text: str) -> bool:
        """유효한 텍스트인지 확인 (한글, 영문, 숫자 포함)"""
        return re.search(r"[가-힣a-zA-Z0-9]", text) is not None
    
    @staticmethod
    def expand_bbox(xs: List, ys: List, img_w: int, img_h: int, ratio: float = 0.3) -> Tuple[int, int, int, int]:
        """Bounding box 확장"""
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        w = x2 - x1
        h = y2 - y1
        x1_exp = max(0, int(x1 - w * ratio))
        x2_exp = min(img_w, int(x2 + w * ratio))
        y1_exp = max(0, int(y1 - h * ratio))
        y2_exp = min(img_h, int(y2 + h * ratio))
        return x1_exp, x2_exp, y1_exp, y2_exp
    
    def vision_ocr_optimized(self, img: np.ndarray) -> Tuple[List, set]:
        """최적화된 OCR (한 번의 API 호출로 모든 텍스트 추출)"""
        _, img_bytes = cv2.imencode(".jpg", img)
        content = img_bytes.tobytes()
        image = vision.Image(content=content)
        
        with self.lock:  # API 호출 동기화
            response = self.client.text_detection(image=image)
        
        texts = response.text_annotations
        results = []
        valid_texts = set()
        
        if texts:
            for text in texts[1:]:
                if self.is_valid_text(text.description.strip()):
                    valid_texts.add(text.description.strip())
                    box = [(v.x, v.y) for v in text.bounding_poly.vertices]
                    while len(box) < 4:
                        box.append(box[-1])
                    results.append((box, text.description.strip()))
        
        return results, valid_texts
    
    def process_single_image(self, img_path: str) -> Dict:
        """단일 이미지에서 텍스트 비율 추출"""
        try:
            orig_img = cv2.imread(img_path)
            if orig_img is None:
                return {"video_id": Path(img_path).stem[:11], "text_ratio": 0.0, "error": "이미지 로드 실패"}
            
            # 이미지 전처리 (2배 확대)
            img_color = cv2.resize(orig_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            img_h, img_w = img_color.shape[:2]
            
            # OCR 수행
            all_results, valid_texts = self.vision_ocr_optimized(img_color)
            
            # 마스크 생성
            mask = np.zeros(img_color.shape[:2], dtype=np.uint8)
            
            for bbox, text in all_results:
                if len(text) == 0 or not self.is_valid_text(text):
                    continue
                pts = np.array(bbox, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
            
            # 텍스트 비율 계산
            text_pixels = cv2.countNonZero(mask)
            total_pixels = mask.shape[0] * mask.shape[1]
            text_ratio = (text_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            
            return {
                "video_id": Path(img_path).stem[:11],
                "text_ratio": round(text_ratio, 2)
            }
        except Exception as e:
            return {"video_id": Path(img_path).stem[:11], "text_ratio": 0.0, "error": str(e)}
    
    def extract_batch(self, image_folder: str, output_csv: str) -> pd.DataFrame:
        """배치로 이미지 폴더 처리"""
        image_files = list(Path(image_folder).glob("*.jpg")) + list(Path(image_folder).glob("*.png"))
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_single_image, str(img)) for img in image_files]
            
            for future in tqdm(futures, desc="📝 텍스트 비율 추출"):
                result = future.result()
                if "error" not in result or result.get("text_ratio", 0) > 0:
                    results.append(result)
        
        df = pd.DataFrame(results)
        save_csv_safely(df, output_csv)
        print(f"✅ 텍스트 피처 추출 완료: {len(df)}개")
        return df


# ========================================
# 2. 색상 클러스터 피처 추출
# ========================================

class ThumbnailColorExtractor:
    """
    썸네일 색상을 CSS3 45개 클러스터로 분류하고 면적 비율 계산
    """
    
    def __init__(self, n_clusters: int = 45):
        """
        Args:
            n_clusters: 색상 클러스터 개수 (기본 45개)
        """
        self.n_clusters = n_clusters
        self.kmeans_model = None
        self.rgb_to_cluster_cache = {}
        self._initialize_color_clusters()
    
    def _initialize_color_clusters(self):
        """CSS3 색상을 LAB 공간에서 K-Means 클러스터링"""
        # CSS3 색상 추출
        css3_names = list(webcolors.CSS3_NAMES_TO_HEX.keys())
        css3_hex_codes = list(webcolors.CSS3_NAMES_TO_HEX.values())
        
        # HEX → RGB 변환
        def hex_to_rgb(hx):
            return tuple(int(hx[i:i+2], 16) for i in (1, 3, 5))
        
        css3_rgb = [hex_to_rgb(hx) for hx in css3_hex_codes]
        css3_rgb_np = np.array(css3_rgb, dtype=np.uint8).reshape(-1, 1, 3)
        
        # RGB → LAB 변환
        css3_lab_np = cv2.cvtColor(css3_rgb_np, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        
        # K-Means 클러스터링
        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        css3_labels = self.kmeans_model.fit_predict(css3_lab_np)
        
        # 수동 클러스터 조정 (학회 논문 기준)
        manual_assignments = {
            'steelblue': 27, 'rosybrown': 39, 'darkkhaki': 30,
            'aquamarine': 14, 'paleturquoise': 22, 'thistle': 44,
            'cadetblue': 22, 'gray': 29, 'grey': 29,
            'lightsteelblue': 9, 'indigo': 38, 'mistyrose': 43
        }
        
        name_to_index = {name: idx for idx, name in enumerate(css3_names)}
        for color_name, new_cluster in manual_assignments.items():
            idx = name_to_index[color_name]
            css3_labels[idx] = new_cluster
        
        # 클러스터 중심 재계산
        css3_centers_lab = self.kmeans_model.cluster_centers_.astype(np.float32)
        target_clusters = sorted(set(manual_assignments.values()))
        
        for cluster_id in target_clusters:
            member_idxs = np.where(css3_labels == cluster_id)[0]
            member_labs = css3_lab_np[member_idxs]
            css3_centers_lab[cluster_id] = member_labs.mean(axis=0)
        
        self.kmeans_model.cluster_centers_ = css3_centers_lab
        print(f"✓ 색상 클러스터 초기화 완료: {self.n_clusters}개")
    
    def get_cluster_id_from_rgb(self, rgb: Tuple[int, int, int]) -> int:
        """RGB 픽셀값을 클러스터 ID로 변환 (캐싱 사용)"""
        if rgb in self.rgb_to_cluster_cache:
            return self.rgb_to_cluster_cache[rgb]
        
        rgb_arr = np.array([[list(rgb)]], dtype=np.uint8)
        lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        cluster_id = int(self.kmeans_model.predict(lab)[0])
        self.rgb_to_cluster_cache[rgb] = cluster_id
        return cluster_id
    
    def extract_color_ratios(self, img_path: str) -> Dict:
        """
        이미지에서 색상 클러스터별 면적 비율 추출
        
        Returns:
            - colorsCluster_0 ~ colorsCluster_44 (45개 컬럼)
            - total_colors: 등장한 색상 클러스터 개수
        """
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("이미지 로드 실패")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            total_pixels = h * w
            cluster_counts = np.zeros(self.n_clusters, dtype=int)
            
            # 모든 픽셀에 대해 클러스터 할당
            for y in range(h):
                for x in range(w):
                    rgb = tuple(img[y, x])
                    cluster_id = self.get_cluster_id_from_rgb(rgb)
                    cluster_counts[cluster_id] += 1
            
            # 비율 계산 (소수점 넷째자리)
            cluster_ratios = cluster_counts / total_pixels
            
            result = {"video_id": Path(img_path).stem[:11]}
            for i in range(self.n_clusters):
                result[f"colorsCluster_{i}"] = round(cluster_ratios[i], 4)
            
            # 총 색상 수 (비율이 0보다 큰 클러스터 개수)
            result["total_colors"] = int(np.sum(cluster_ratios > 0))
            
            return result
        except Exception as e:
            print(f"❌ 색상 추출 실패 ({img_path}): {e}")
            return None
    
    def extract_batch(self, image_folder: str, output_csv: str) -> pd.DataFrame:
        """배치 처리"""
        image_files = list(Path(image_folder).glob("*.jpg")) + list(Path(image_folder).glob("*.png"))
        
        results = []
        for img_path in tqdm(image_files, desc="🎨 색상 클러스터 분석"):
            result = self.extract_color_ratios(str(img_path))
            if result:
                results.append(result)
        
        df = pd.DataFrame(results)
        save_csv_safely(df, output_csv)
        print(f"✅ 색상 피처 추출 완료: {len(df)}개")
        return df


# ========================================
# 3-6. 기타 피처 (밝기, 질감, 객체, 테마 매칭)
# ========================================

class ThumbnailVisualExtractor:
    """
    밝기 대비, 질감 선명도 등의 시각적 피처 추출
    + 표준화 스케일링 포함
    """
    
    @staticmethod
    def extract_brightness_weighted_std(
        colors_df: pd.DataFrame, 
        meta_csv_path: str
    ) -> pd.DataFrame:
        """
        색상 클러스터별 명암(밝기) 가중 표준편차 계산
        + StandardScaler + Sigmoid 함수 적용
        
        Args:
            colors_df: thumbnails_colorsRatio.csv 데이터프레임 (colorsCluster_0~44 포함)
            meta_csv_path: colorsCluster_meta.csv 경로 (V_hsv 값 포함)
            
        Returns:
            brightness_weightedStd, brightness_weightedStd_scaledSigmoid 컬럼 추가된 DataFrame
        """
        # 메타 데이터 로드
        df_meta = pd.read_csv(meta_csv_path)
        V_vals = df_meta['V_hsv'].values
        
        def weighted_std(x, weights):
            """가중 표준편차 계산"""
            x = np.array(x)
            weights = np.array(weights)
            average = np.sum(weights * x) / np.sum(weights)
            variance = np.sum(weights * (x - average) ** 2) / np.sum(weights)
            return np.sqrt(variance)
        
        # 클러스터 컬럼명
        ratio_cols = [f'colorsCluster_{i}' for i in range(len(V_vals))]
        
        # 가중 표준편차 계산
        brightness_weighted_std = colors_df[ratio_cols].apply(
            lambda row: weighted_std(V_vals, row.values),
            axis=1
        )
        
        # StandardScaler + Sigmoid 함수 적용
        scaler = StandardScaler()
        std_scaled = scaler.fit_transform(brightness_weighted_std.values.reshape(-1, 1))
        alpha = 1
        sigmoid_vals = 1 / (1 + np.exp(-alpha * std_scaled))
        
        result_df = colors_df[['video_id']].copy()
        result_df['brightness_weightedStd'] = brightness_weighted_std.round(4)
        result_df['brightness_weightedStd_scaledSigmoid'] = sigmoid_vals.flatten().round(4)
        
        return result_df
    
    @staticmethod
    def extract_texture_sharpness(img_path: str) -> Dict:
        """질감 및 선명도 추출 (Laplacian variance)"""
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("이미지 로드 실패")
            
            # Laplacian을 이용한 선명도 측정
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            
            return {
                "video_id": Path(img_path).stem[:11],
                "texture_sharpness": round(laplacian_var, 2)
            }
        except Exception as e:
            return {"video_id": Path(img_path).stem[:11], "texture_sharpness": 0.0}
    
    def extract_batch_texture(self, image_folder: str, output_csv: str) -> pd.DataFrame:
        """
        질감 피처 배치 추출 + StandardScaler 적용
        
        Returns:
            texture_sharpness, texture_sharpness_scaled 컬럼 포함
        """
        image_files = list(Path(image_folder).glob("*.jpg")) + list(Path(image_folder).glob("*.png"))
        
        results = []
        for img_path in tqdm(image_files, desc="🌀 텍스처 분석"):
            results.append(self.extract_texture_sharpness(str(img_path)))
        
        df = pd.DataFrame(results)
        
        # StandardScaler 적용
        if 'texture_sharpness' in df.columns:
            scaler = StandardScaler()
            df['texture_sharpness_scaled'] = scaler.fit_transform(
                df[['texture_sharpness']]
            ).flatten().round(4)
        
        save_csv_safely(df, output_csv)
        print(f"✅ 질감 피처 추출 완료: {len(df)}개")
        
        return df


# ========================================
# 4. 색상 테마 매칭 피처
# ========================================

class ThumbnailColorThemeExtractor:
    """
    썸네일 색상과 일상/감성 테마 매칭 점수 계산
    """
    
    def __init__(self, color_feeling_map_path: str):
        """
        Args:
            color_feeling_map_path: colorsFeelingMatch_map_dailySensScores.csv 경로
        """
        self.color_feeling_map = pd.read_csv(color_feeling_map_path)
        self._prepare_lab_table()
    
    def _prepare_lab_table(self):
        """RGB를 LAB으로 변환하여 테이블 준비"""
        def rgb_to_lab_row(row):
            rgb = np.uint8([[[row['R'], row['G'], row['B']]]])
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0][0]
            return pd.Series({'L_lab': lab[0], 'A_lab': lab[1], 'B_lab': lab[2]})
        
        self.color_feeling_map[['L_lab', 'A_lab', 'B_lab']] = self.color_feeling_map.apply(
            rgb_to_lab_row, axis=1
        )
        
        self.lab_values = self.color_feeling_map[['L_lab', 'A_lab', 'B_lab']].values.astype(np.float32)
        self.score_daily_vals = self.color_feeling_map['daily_score'].values
        self.score_sens_vals = self.color_feeling_map['sensibility_score'].values
        self.lab_score_cache = {}
    
    def compute_matching_scores_with_cache(self, img_lab: np.ndarray) -> Tuple[float, float]:
        """
        이미지의 모든 픽셀에 대해 가장 가까운 색상의 일상/감성 점수 평균 계산
        
        Returns:
            (daily_matchScore, sensibility_matchScore)
        """
        H, W, _ = img_lab.shape
        flattened = img_lab.reshape(-1, 3)
        
        matched_daily = []
        matched_sens = []
        
        for pix_lab in flattened:
            pix_key = tuple(pix_lab)
            
            if pix_key in self.lab_score_cache:
                daily, sens = self.lab_score_cache[pix_key]
            else:
                dists = np.linalg.norm(self.lab_values - pix_lab, axis=1)
                idx = np.argmin(dists)
                daily = self.score_daily_vals[idx]
                sens = self.score_sens_vals[idx]
                self.lab_score_cache[pix_key] = (daily, sens)
            
            matched_daily.append(daily)
            matched_sens.append(sens)
        
        return np.mean(matched_daily), np.mean(matched_sens)
    
    def extract_theme_scores(self, img_path: str) -> Dict:
        """
        단일 이미지에서 색상 테마 매칭 점수 추출
        
        Returns:
            - colorsDaily_matchScore
            - colorsSensibility_matchScore
            - colorsTheme_matchScore (둘 중 큰 값)
        """
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("이미지 로드 실패")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            
            score_d, score_s = self.compute_matching_scores_with_cache(img_lab)
            
            return {
                'video_id': Path(img_path).stem[:11],
                'colorsDaily_matchScore': round(score_d, 4),
                'colorsSensibility_matchScore': round(score_s, 4),
                'colorsTheme_matchScore': round(max(score_d, score_s), 4)
            }
        except Exception as e:
            print(f"❌ 테마 매칭 실패 ({img_path}): {e}")
            return None
    
    def extract_batch(self, image_folder: str, output_csv: str) -> pd.DataFrame:
        """배치 처리"""
        image_files = list(Path(image_folder).glob("*.jpg")) + list(Path(image_folder).glob("*.png"))
        
        results = []
        for img_path in tqdm(image_files, desc="🎨 색상 테마 매칭"):
            result = self.extract_theme_scores(str(img_path))
            if result:
                results.append(result)
        
        df = pd.DataFrame(results)
        save_csv_safely(df, output_csv)
        print(f"✅ 색상 테마 매칭 완료: {len(df)}개")
        
        return df


# ========================================
# 메인 통합 클래스
# ========================================

class ThumbnailFeatureExtractor:
    """
    모든 썸네일 피처를 통합 관리하는 메인 클래스
    """
    
    def __init__(self, google_credentials_path: Optional[str] = None):
        """
        Args:
            google_credentials_path: Google Cloud Vision API 인증 파일 (텍스트 추출용)
        """
        self.google_credentials_path = google_credentials_path
        self.gpu_info = setup_gpu()
        print(f"✓ GPU 설정: {self.gpu_info['device']}")
    
    def extract_all_features(self, 
                            image_folder: str, 
                            output_dir: str,
                            extract_text: bool = True,
                            extract_colors: bool = True,
                            extract_visual: bool = True):
        """
        모든 썸네일 피처를 한 번에 추출
        
        Args:
            image_folder: 썸네일 이미지 폴더
            output_dir: 결과 CSV 저장 디렉토리
            extract_text: 텍스트 비율 추출 여부
            extract_colors: 색상 클러스터 추출 여부
            extract_visual: 밝기/질감 추출 여부
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        # 1. 텍스트 비율
        if extract_text and self.google_credentials_path:
            print("\n[1/3] 텍스트 비율 추출 시작...")
            text_extractor = ThumbnailTextExtractor(self.google_credentials_path)
            results['text'] = text_extractor.extract_batch(
                image_folder, 
                os.path.join(output_dir, "thumbnails_text.csv")
            )
        
        # 2. 색상 클러스터
        if extract_colors:
            print("\n[2/3] 색상 클러스터 추출 시작...")
            color_extractor = ThumbnailColorExtractor()
            results['colors'] = color_extractor.extract_batch(
                image_folder,
                os.path.join(output_dir, "thumbnails_colorsRatio.csv")
            )
        
        # 3. 밝기/질감
        if extract_visual:
            print("\n[3/3] 밝기/질감 추출 시작...")
            visual_extractor = ThumbnailVisualExtractor()
            
            # 질감 피처 추출
            texture_df = visual_extractor.extract_batch_texture(
                image_folder,
                os.path.join(output_dir, "thumbnails_textureSharpness.csv")
            )
            results['texture'] = texture_df
            
            print("  ⚠️  밝기 피처는 색상 클러스터 데이터가 필요합니다.")
            print("  → 색상 피처 추출 후 별도로 실행하세요:")
            print("     brightness_df = ThumbnailVisualExtractor.extract_brightness_weighted_std(")
            print("         colors_df, 'path/to/colorsCluster_meta.csv')")
        
        print(f"\n✅ 모든 썸네일 피처 추출 완료!")
        print(f"📁 결과 저장 위치: {output_dir}")
        
        return results


# ========================================
# 사용 예시
# ========================================

if __name__ == "__main__":
    # 설정
    IMAGE_FOLDER = "../thumbnails_image/raw_thumbnails"
    OUTPUT_DIR = "../rawData/thumbnails"
    # Google Cloud 인증 파일 경로 (환경변수 사용 권장)
    GOOGLE_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "./credentials/google-vision-api.json")
    
    # 통합 추출
    extractor = ThumbnailFeatureExtractor(google_credentials_path=GOOGLE_CREDENTIALS)
    
    results = extractor.extract_all_features(
        image_folder=IMAGE_FOLDER,
        output_dir=OUTPUT_DIR,
        extract_text=True,
        extract_colors=True,
        extract_visual=True
    )
    
    print("\n📊 추출된 피처:")
    for key, df in results.items():
        print(f"  - {key}: {len(df)} rows")
