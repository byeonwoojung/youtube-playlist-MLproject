"""
데이터 병합 모듈

모든 피처 CSV 파일들을 병합하여 최종 데이터프레임 생성

피처 카테고리:
- 썸네일 피처: 텍스트, 색상, 밝기, 질감, 얼굴, 객체 등
- 오디오 감정 피처 (Audio Emotional): happy, sad, angry, fear, surprise, disgust, neutral
- 오디오 음악 피처 (Audio Quantitative): BPM, 피치, 에너지 등
- 제목 피처: 이모지, 해시태그, 문자 길이 등
- 메타 피처: 구독자 수, 업로드 날짜, 콘텐츠 타입 등
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict

import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.helpers import load_csv_safely, save_csv_safely
except ImportError:
    def load_csv_safely(filepath, encoding="utf-8"):
        return pd.read_csv(filepath, encoding=encoding, float_precision='round_trip')
    
    def save_csv_safely(df, filepath, encoding="utf-8-sig"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False, encoding=encoding)
        return True


def merge_all_features(
    base_info_path: str,
    thumbnail_dir: str,
    audio_dir: str,
    titles_path: str,
    output_path: str
) -> pd.DataFrame:
    """
    모든 피처 데이터를 병합
    
    Args:
        base_info_path: 기본 YouTube 정보 CSV 경로
        thumbnail_dir: 썸네일 피처 CSV들이 있는 디렉토리
        audio_dir: 오디오 피처 CSV들이 있는 디렉토리
        titles_path: 제목 피처 CSV 경로
        output_path: 병합 결과 저장 경로
    
    Returns:
        병합된 DataFrame
    """
    
    print("=" * 60)
    print("데이터 병합 시작".center(60))
    print("=" * 60)
    
    # 1. 기본 정보 로드
    print("\n[1/4] 기본 YouTube 정보 로드...")
    df_base = load_csv_safely(base_info_path)
    df_base = df_base[[
        'video_id', 'publish_date',
        'subscriber_count', 'views',
        'sensibility', 'daily', 'hashtags'
    ]]
    print(f"  ✓ Base info: {len(df_base)} rows")
    
    # 2. 썸네일 피처 로드
    print("\n[2/4] 썸네일 피처 로드...")
    thumbnail_files = {
        'face': 'thumbnails_face.csv',
        'colors': 'thumbnails_colorsRatio.csv',
        'brightness': 'thumbnails_colorsBrightness.csv',  # 또는 thumbnails_brightness.csv
        'theme': 'thumbnails_colorsThemeMatch.csv',
        'objects': 'thumbnails_objects.csv',
        'text': 'thumbnails_text.csv',
        'texture': 'thumbnails_textureSharpness.csv'  # 또는 thumbnails_texture.csv
    }
    
    df = df_base.copy()
    
    for key, filename in thumbnail_files.items():
        file_path = os.path.join(thumbnail_dir, filename)
        if os.path.exists(file_path):
            df_feature = load_csv_safely(file_path)
            
            # 특정 컬럼 제거
            if key == 'face' and 'image_count' in df_feature.columns:
                df_feature = df_feature.drop(columns=['image_count'])
            
            if key == 'objects':
                df_feature = df_feature[['video_id', 'person', 'animal', 'anime', 'landscape']]
            
            df = pd.merge(df, df_feature, how='inner', on='video_id')
            print(f"  ✓ {key}: {len(df)} rows (병합 후)")
        else:
            print(f"  ⚠ {key}: 파일 없음 ({filename})")
    
    # 3. 오디오 피처 로드
    print("\n[3/4] 오디오 피처 로드...")
    
    # 정성적 피처
    audio_qualitative_path = os.path.join(audio_dir, 'audio_qualitative.csv')
    if os.path.exists(audio_qualitative_path):
        df_audio_qual = load_csv_safely(audio_qualitative_path)
        # 불필요한 컬럼 제거
        drop_cols = ['pitch_mean', 'energy_mean', 'centroid_mean', 'bmp', 
                    'speech_rate', 'initial_silence', 'emotion_name', 'confidence']
        df_audio_qual = df_audio_qual.drop(columns=[c for c in drop_cols if c in df_audio_qual.columns])
        df = pd.merge(df, df_audio_qual, how='inner', on='video_id')
        print(f"  ✓ Audio qualitative: {len(df)} rows")
    
    # 정량적 피처
    audio_quantitative_path = os.path.join(audio_dir, 'audio_quantitative_retry.csv')
    # 또는 audio_quantitative.csv
    if not os.path.exists(audio_quantitative_path):
        audio_quantitative_path = os.path.join(audio_dir, 'audio_quantitative.csv')
    
    if os.path.exists(audio_quantitative_path):
        df_audio_quant = load_csv_safely(audio_quantitative_path)
        df_audio_quant = df_audio_quant.drop(columns=['error', 'url'], errors='ignore')
        df = pd.merge(df, df_audio_quant, how='inner', on='video_id')
        print(f"  ✓ Audio quantitative: {len(df)} rows")
    
    # 4. 제목 피처 로드
    print("\n[4/4] 제목 피처 로드...")
    if os.path.exists(titles_path):
        df_titles = load_csv_safely(titles_path)
        # has_emoji를 int로 변환
        if 'has_emoji' in df_titles.columns:
            df_titles['has_emoji'] = df_titles['has_emoji'].astype(int)
        df = pd.merge(df, df_titles, how='inner', on='video_id')
        print(f"  ✓ Titles: {len(df)} rows")
    
    # 5. 소수점 반올림 (float 컬럼만)
    print("\n[5/5] 데이터 정리...")
    float_columns = df.select_dtypes(include='float').columns
    df[float_columns] = df[float_columns].round(4)
    
    # 6. 저장
    save_csv_safely(df, output_path)
    
    print("\n" + "=" * 60)
    print(f"✅ 병합 완료: {len(df)} rows, {len(df.columns)} columns")
    print(f"📁 저장 위치: {output_path}")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    # 예시 실행
    BASE_INFO = "../rawData/youtubeInfo/allYoutubeInfo_themeFiltered.csv"
    THUMBNAIL_DIR = "../rawData/thumbnails"
    AUDIO_DIR = "../rawData/audio"
    TITLES = "../rawData/titles/titles_final.csv"
    OUTPUT = "../rawData/data_merged.csv"
    
    df_merged = merge_all_features(
        base_info_path=BASE_INFO,
        thumbnail_dir=THUMBNAIL_DIR,
        audio_dir=AUDIO_DIR,
        titles_path=TITLES,
        output_path=OUTPUT
    )
    
    print("\n📊 병합된 데이터 미리보기:")
    print(df_merged.info())
