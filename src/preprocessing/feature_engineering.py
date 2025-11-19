"""
피처 엔지니어링 모듈

병합된 데이터에서 추가 피처 생성:
- hashtag_count: 해시태그 개수
- 시간대 원핫인코딩 (새벽, 아침, 낮, 저녁, 밤)
- time_match_content: 시간대와 콘텐츠 매칭 여부
- text_char_combo: 텍스트 면적 * 제목 길이
- object_complexity: 객체 요소 복잡도

Target Variable: log_views (로그 변환된 조회수)
평가 지표: RMSE (Root Mean Squared Error)
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional

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


def engineer_features(input_csv: str, output_csv: str) -> pd.DataFrame:
    """
    피처 엔지니어링 수행
    
    Args:
        input_csv: 병합된 데이터 CSV 경로
        output_csv: 최종 데이터 저장 경로
    
    Returns:
        피처 엔지니어링이 완료된 DataFrame
    """
    
    print("=" * 60)
    print("피처 엔지니어링 시작".center(60))
    print("=" * 60)
    
    # 데이터 로드
    print(f"\n[1/5] 데이터 로드: {input_csv}")
    df = load_csv_safely(input_csv)
    print(f"  ✓ 로드 완료: {len(df)} rows, {len(df.columns)} columns")
    
    # 1. hashtag_count 생성
    print("\n[2/5] hashtag_count 생성...")
    df['hashtag_count'] = df['hashtags'].fillna('').apply(
        lambda x: len([tag.strip() for tag in x.split(',') if tag.strip()])
    )
    print(f"  ✓ hashtag_count 추가 (평균: {df['hashtag_count'].mean():.2f})")
    
    # 2. 업로드 시간대 원핫인코딩
    print("\n[3/5] 업로드 시간대 원핫인코딩...")
    
    # datetime 변환
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    df['hour'] = df['publish_date'].dt.hour
    
    # 시간대 구분 함수
    def get_time_period(hour):
        """
        시간대 분류:
        - midnight: 0-6시 (새벽)
        - morning: 6-10시 (아침)
        - noon: 10-16시 (낮)
        - evening: 16-21시 (저녁)
        - night: 21-24시 (밤)
        """
        if 0 <= hour < 6:
            return 'midnight'
        elif 6 <= hour < 10:
            return 'morning'
        elif 10 <= hour < 16:
            return 'noon'
        elif 16 <= hour < 21:
            return 'evening'
        else:
            return 'night'
    
    df['time_period'] = df['hour'].apply(get_time_period)
    
    # 원핫인코딩
    df = pd.get_dummies(df, columns=['time_period'], prefix='time')
    
    # hour 컬럼 제거
    df.drop(columns=['hour'], inplace=True)
    
    print(f"  ✓ 시간대 원핫인코딩 완료")
    print(f"    - time_midnight: {df.get('time_midnight', pd.Series([0])).sum()} videos")
    print(f"    - time_morning: {df.get('time_morning', pd.Series([0])).sum()} videos")
    print(f"    - time_noon: {df.get('time_noon', pd.Series([0])).sum()} videos")
    print(f"    - time_evening: {df.get('time_evening', pd.Series([0])).sum()} videos")
    print(f"    - time_night: {df.get('time_night', pd.Series([0])).sum()} videos")
    
    # 3. 시간대와 감성/일상 콘텐츠 매칭
    print("\n[4/5] 시간대-콘텐츠 매칭 피처 생성...")
    
    def match_time_content(row):
        """
        시간대와 콘텐츠 매칭 여부:
        - 새벽/저녁/밤 + 감성 콘텐츠 = 매치
        - 아침/낮/저녁 + 일상 콘텐츠 = 매치
        """
        # 감성 콘텐츠 매칭
        if ((row.get('time_midnight', 0) == 1 or 
             row.get('time_evening', 0) == 1 or 
             row.get('time_night', 0) == 1) and 
            row.get('sensibility', 0) == 1):
            return 1
        
        # 일상 콘텐츠 매칭
        if ((row.get('time_morning', 0) == 1 or 
             row.get('time_noon', 0) == 1 or 
             row.get('time_evening', 0) == 1) and 
            row.get('daily', 0) == 1):
            return 1
        
        return 0
    
    df['time_match_content'] = df.apply(match_time_content, axis=1)
    matched_count = df['time_match_content'].sum()
    print(f"  ✓ time_match_content 추가 (매칭: {matched_count}/{len(df)} = {matched_count/len(df)*100:.1f}%)")
    
    # 4. 텍스트 효과 및 객체 복잡도
    print("\n[5/5] 복합 피처 생성...")
    
    # 텍스트 효과 (썸네일 텍스트 비율 * 제목 길이)
    if 'text_ratio' in df.columns and 'char_length' in df.columns:
        df['text_char_combo'] = df['text_ratio'] * df['char_length']
        print(f"  ✓ text_char_combo 추가 (평균: {df['text_char_combo'].mean():.2f})")
    
    # 객체 복잡도 (사람 + 동물 + 애니메이션 + 풍경)
    object_columns = ['person', 'animal', 'anime', 'landscape']
    if all(col in df.columns for col in object_columns):
        df['object_complexity'] = df[object_columns].sum(axis=1)
        print(f"  ✓ object_complexity 추가 (평균: {df['object_complexity'].mean():.2f})")
    
    # 5. 저장
    save_csv_safely(df, output_csv)
    
    print("\n" + "=" * 60)
    print(f"✅ 피처 엔지니어링 완료")
    print(f"   총 {len(df)} rows, {len(df.columns)} columns")
    print(f"📁 저장 위치: {output_csv}")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    # 예시 실행
    INPUT_CSV = "../rawData/data_merged.csv"
    OUTPUT_CSV = "../rawData/final_data.csv"
    
    df_final = engineer_features(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV
    )
    
    print("\n📊 최종 데이터 정보:")
    print(df_final.info())
    
    print("\n📈 추가된 피처들:")
    new_features = ['hashtag_count', 'time_midnight', 'time_morning', 'time_noon', 
                   'time_evening', 'time_night', 'time_match_content', 
                   'text_char_combo', 'object_complexity']
    for feature in new_features:
        if feature in df_final.columns:
            print(f"  ✓ {feature}")
