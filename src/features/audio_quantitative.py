"""
오디오 정량적 피처 추출 모듈

YouTube URL에서 오디오를 다운로드하고 정량적 피처를 추출합니다:
- BPM (템포)
- Pitch (음높이) 
- Energy (에너지)
- Spectral Centroid (음색 밝기)
- Speech Rate (발화 속도)
- Initial Silence (초기 무음)
"""

import os
import subprocess
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import ast

# 환경 설정
os.environ["PATH"] = "/opt/homebrew/opt/ffmpeg/bin:" + os.environ["PATH"]
SAVE_DIR = "temp_audio"
TEMP_DIR = "../rawData/audio/tempAudio"
OUTPUT_FILE = "../rawData/audio/audio_quantitative.csv"
RETRY_OUTPUT_FILE = "../rawData/audio/audio_quantitative_errorRetry.csv"
FINAL_OUTPUT_FILE = "../rawData/audio/audio_quantitative_retry.csv"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


# ========================================
# 유틸리티 함수
# ========================================

def extract_video_id(url):
    """YouTube URL에서 video_id 추출"""
    parsed = urlparse(url)
    return parse_qs(parsed.query).get("v", [None])[0]


def load_audio_fast_ffmpeg(path, sr=22050):
    """FFmpeg를 사용한 빠른 오디오 로딩"""
    command = ["ffmpeg", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
    return np.frombuffer(process.stdout, np.float32), sr


def get_audio_path(video_id):
    """오디오 파일 경로 생성"""
    return os.path.join(SAVE_DIR, f"{video_id}.m4a")


# ========================================
# 피처 추출 함수
# ========================================

def process_video(url):
    """
    단일 YouTube 비디오 처리 (다운로드 + 피처 추출)
    """
    video_id = extract_video_id(url)
    result = {
        "video_id": video_id, "url": url,
        "pitch_mean": None, "energy_mean": None, "centroid_mean": None,
        "bpm": None, "speech_rate": None, "initial_silence": None, "error": None
    }
    output_path = get_audio_path(video_id)

    try:
        command = [
            "yt-dlp", url, "-f", "bestaudio[ext=m4a]",
            "--download-sections", "*0-60", "--user-agent", "Mozilla/5.0",
            "--socket-timeout", "10", "--retries", "2",
            "-o", f"{SAVE_DIR}/{video_id}.%(ext)s"
        ]
        proc = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if not os.path.exists(output_path):
            raise FileNotFoundError("m4a 다운로드 실패")

        y, sr = load_audio_fast_ffmpeg(output_path)

        # Pitch (음높이)
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        result["pitch_mean"] = np.nanmean(f0)
        
        # Energy (에너지)
        rms = librosa.feature.rms(y=y)
        result["energy_mean"] = np.mean(rms)
        
        # Spectral Centroid (음색 밝기)
        result["centroid_mean"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # BPM (템포) - 리스트 형태로 반환됨
        result["bpm"] = librosa.beat.beat_track(y=y, sr=sr)[0]
        
        # Speech Rate (발화 속도)
        result["speech_rate"] = len(librosa.onset.onset_detect(y=y, sr=sr, units='time'))
        
        # Initial Silence (초기 무음)
        rms_vals = rms[0]
        result["initial_silence"] = librosa.frames_to_time(np.argmax(rms_vals > 0.01), sr=sr) if np.any(rms_vals > 0.01) else 60.0

    except subprocess.CalledProcessError as e:
        result["error"] = f"yt-dlp error: {e.stderr.decode(errors='ignore')[:300]}"
    except Exception as e:
        result["error"] = str(e)
    finally:
        try:
            os.remove(output_path)
        except:
            pass

    return result


# ========================================
# 1단계: 전체 영상 처리
# ========================================

def main():
    """
    전체 YouTube URL 처리
    """
    df = pd.read_csv('../rawData/youtubeInfo/allYoutubeInfo_themeFiltered.csv')
    video_data = [url for url in df["video_url"] if extract_video_id(url)]

    if os.path.exists(OUTPUT_FILE):
        processed_ids = set(pd.read_csv(OUTPUT_FILE)["video_id"])
    else:
        processed_ids = set()

    start = time.time()
    batch_results = []
    batch_count = 0

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_video, url)
            for url in video_data if extract_video_id(url) not in processed_ids
        ]
        for i, f in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing videos", dynamic_ncols=True, leave=True)):
            try:
                result = f.result()
                batch_results.append(result)

                if len(batch_results) >= 100:
                    temp_path = os.path.join(TEMP_DIR, f"batch_{batch_count:04d}.csv")
                    pd.DataFrame(batch_results).to_csv(
                        temp_path, index=False,
                        encoding="utf-8-sig", float_format="%.4f"
                    )
                    batch_results.clear()
                    batch_count += 1
            except Exception as e:
                print(f"[ERROR] {e}")

    if batch_results:
        temp_path = os.path.join(TEMP_DIR, f"batch_{batch_count:04d}.csv")
        pd.DataFrame(batch_results).to_csv(
            temp_path, index=False,
            encoding="utf-8-sig", float_format="%.4f"
        )

    print(f"\n⏱ 전체 소요 시간: {time.time() - start:.2f}초")


def merge_temp_batches():
    """
    배치 파일 병합
    """
    import glob

    temp_files = sorted(glob.glob(os.path.join(TEMP_DIR, "batch_*.csv")))
    if not temp_files:
        print("⚠️ 병합할 batch_*.csv 파일이 없습니다.")
        return

    df_all = pd.concat([pd.read_csv(f) for f in temp_files], ignore_index=True)
    df_all.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig", float_format="%.4f")
    print(f"✅ 병합 완료 → {OUTPUT_FILE} 저장됨")



# ========================================
# 2단계: 에러 재시도
# ========================================

def retry_error_videos():
    """
    에러 발생한 영상만 재시도
    """
    df = pd.read_csv(OUTPUT_FILE)
    error_rows = df[df["error"].notnull()]

    print(f"🔁 재처리할 영상 수: {len(error_rows)}")

    results = []
    for _, row in tqdm(error_rows.iterrows(), total=len(error_rows), desc="Retrying errors"):
        results.append(process_video(row["url"]))

    df_retry = pd.DataFrame(results)
    df_retry.to_csv(RETRY_OUTPUT_FILE, index=False, encoding="utf-8-sig", float_format="%.4f")
    print(f"✅ 재처리 결과 저장 완료 → {RETRY_OUTPUT_FILE}")



# ========================================
# 3단계: 후처리
# ========================================

def postprocess_audio():
    """
    오디오 정량적 피처 후처리
    
    1. 원본 + 재시도 CSV 병합
    2. BPM 리스트 문자열 → 숫자 변환
    3. 결측치 제거
    """
    # 병합
    df_original = pd.read_csv(OUTPUT_FILE)
    df_retry = pd.read_csv(RETRY_OUTPUT_FILE)

    # retry 결과에서 error가 없는 데이터만 유지
    df_retry_success = df_retry[df_retry["error"].isnull()]

    # video_id 기준으로 기존 데이터 업데이트
    df_updated = df_original.set_index("video_id").combine_first(df_retry_success.set_index("video_id")).reset_index()

    # 후처리
    # 1) bpm: 리스트가 문자열 형태로 된 것 -> 리스트화 시킨 후, 숫자만 가져옴
    # 2) 결측치 있는 행 제거
    
    # bpm 문자열 → 리스트 처리 후 첫 값 추출
    df_updated['bpm'] = df_updated['bpm'].apply(
        lambda x: round(ast.literal_eval(x)[0], 4) if isinstance(x, str) and x.startswith('[') else x
    )

    # 중요 컬럼 정의 (분석에 꼭 필요한 컬럼들)
    important_cols = ['bpm', 'pitch_mean', 'energy_mean', 'centroid_mean', 'speech_rate', 'initial_silence']

    # 중요 컬럼 중 하나라도 결측치가 있으면 제거
    df_cleaned = df_updated.dropna(subset=important_cols)

    # 저장
    df_cleaned.to_csv(FINAL_OUTPUT_FILE, encoding='utf-8-sig', index=False, float_format="%.4f")
    print(f"✅ 결측치 제거 후 저장 완료: {len(df_updated)} → {len(df_cleaned)} 행 유지됨")


# ========================================
# 메인 실행
# ========================================

if __name__ == "__main__":
    # 1단계: 전체 영상 추출
    print("\n[1/3] 전체 영상 오디오 피처 추출...")
    main()
    merge_temp_batches()
    
    # 2단계: 에러 재시도
    print("\n[2/3] 에러 영상 재시도...")
    retry_error_videos()
    
    # 3단계: 후처리 (병합 + BPM 파싱 + 결측치 제거)
    print("\n[3/3] 후처리 (병합 + BPM 파싱 + 결측치 제거)...")
    postprocess_audio()
    
    print("\n" + "=" * 60)
    print("✅ 전체 오디오 처리 완료!")
    print("=" * 60)


