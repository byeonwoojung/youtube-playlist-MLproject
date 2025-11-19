"""
피처 추출 실행 스크립트

모든 피처를 순차적으로 추출합니다:
1. 썸네일 피처 (텍스트, 색상, 밝기, 질감, 얼굴, 객체)
2. 오디오 피처 (정량적, 정성적)
3. 제목 피처
4. 데이터 병합
5. 피처 엔지니어링
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.thumbnail_features import ThumbnailFeatureExtractor
from src.preprocessing.data_merger import merge_all_features
from src.preprocessing.feature_engineering import engineer_features

def main():
    print("=" * 80)
    print("YouTube 조회수 예측 - 피처 추출 파이프라인".center(80))
    print("=" * 80)
    
    # 경로 설정
    BASE_DIR = project_root
    DATA_DIR = BASE_DIR / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    
    # 디렉토리 생성
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "thumbnails").mkdir(exist_ok=True)
    
    # Google Cloud 인증 파일 경로 (환경변수 사용 권장)
    GOOGLE_CREDENTIALS = BASE_DIR / "credentials" / "google-vision-api.json"
    if not GOOGLE_CREDENTIALS.exists():
        GOOGLE_CREDENTIALS = Path(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""))
    
    # ===================================
    # 1. 썸네일 피처 추출
    # ===================================
    print("\n" + "=" * 80)
    print("[1/5] 썸네일 피처 추출 시작".center(80))
    print("=" * 80)
    
    thumbnail_folder = RAW_DIR / "thumbnails"
    
    if thumbnail_folder.exists() and GOOGLE_CREDENTIALS.exists():
        try:
            extractor = ThumbnailFeatureExtractor(
                google_credentials_path=str(GOOGLE_CREDENTIALS)
            )
            
            thumbnail_results = extractor.extract_all_features(
                image_folder=str(thumbnail_folder),
                output_dir=str(PROCESSED_DIR / "thumbnails"),
                extract_text=True,
                extract_colors=True,
                extract_visual=True
            )
            
            print("\n✅ 썸네일 피처 추출 완료!")
        except Exception as e:
            print(f"\n❌ 썸네일 피처 추출 실패: {e}")
    else:
        print(f"\n⚠ 썸네일 폴더 또는 인증 파일을 찾을 수 없습니다.")
        print(f"  - 썸네일 폴더: {thumbnail_folder}")
        print(f"  - 인증 파일: {GOOGLE_CREDENTIALS}")
    
    # ===================================
    # 2. 오디오 피처는 별도 실행 권장
    # ===================================
    print("\n" + "=" * 80)
    print("[2/5] 오디오 피처 (별도 실행 권장)".center(80))
    print("=" * 80)
    print("⚠ 오디오 피처는 시간이 오래 걸리므로 다음 스크립트를 별도로 실행하세요:")
    print("  python src/features/audio_quantitative.py")
    print("  python src/features/audio_qualitative.py")
    
    # ===================================
    # 3. 제목 피처는 노트북에서 실행
    # ===================================
    print("\n" + "=" * 80)
    print("[3/5] 제목 피처 (노트북에서 실행)".center(80))
    print("=" * 80)
    print("⚠ 제목 피처는 원본 6_titles.ipynb를 실행하여 생성하세요.")
    
    # ===================================
    # 4. 데이터 병합
    # ===================================
    print("\n" + "=" * 80)
    print("[4/5] 데이터 병합 시작".center(80))
    print("=" * 80)
    
    base_info_path = RAW_DIR / "youtubeInfo" / "allYoutubeInfo_themeFiltered.csv"
    titles_path = RAW_DIR / "titles" / "titles_final.csv"
    output_merged_path = PROCESSED_DIR / "data_merged.csv"
    
    if base_info_path.exists():
        try:
            df_merged = merge_all_features(
                base_info_path=str(base_info_path),
                thumbnail_dir=str(RAW_DIR / "thumbnails"),  # 또는 PROCESSED_DIR / "thumbnails"
                audio_dir=str(RAW_DIR / "audio"),
                titles_path=str(titles_path),
                output_path=str(output_merged_path)
            )
            print("\n✅ 데이터 병합 완료!")
        except Exception as e:
            print(f"\n❌ 데이터 병합 실패: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠ 기본 정보 파일을 찾을 수 없습니다: {base_info_path}")
    
    # ===================================
    # 5. 피처 엔지니어링
    # ===================================
    print("\n" + "=" * 80)
    print("[5/5] 피처 엔지니어링 시작".center(80))
    print("=" * 80)
    
    if output_merged_path.exists():
        try:
            output_final_path = PROCESSED_DIR / "final_data.csv"
            
            df_final = engineer_features(
                input_csv=str(output_merged_path),
                output_csv=str(output_final_path)
            )
            
            print("\n✅ 피처 엔지니어링 완료!")
            print(f"\n📊 최종 데이터:")
            print(f"  - Rows: {len(df_final)}")
            print(f"  - Columns: {len(df_final.columns)}")
            print(f"  - 저장 위치: {output_final_path}")
            
        except Exception as e:
            print(f"\n❌ 피처 엔지니어링 실패: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠ 병합 파일을 찾을 수 없습니다: {output_merged_path}")
    
    # ===================================
    # 완료
    # ===================================
    print("\n" + "=" * 80)
    print("🎉 피처 추출 파이프라인 완료!".center(80))
    print("=" * 80)
    print("\n다음 단계:")
    print("  1. 오디오 피처가 없다면 별도로 실행")
    print("  2. notebooks/model_analysis.ipynb에서 모델링 시작")
    print("=" * 80)

if __name__ == "__main__":
    main()
