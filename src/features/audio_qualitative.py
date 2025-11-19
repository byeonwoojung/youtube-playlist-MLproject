# Wav2wav2는 원시파형(시간에 따른 진폭)으로 분석
# 오디오 정량적 특성은 기준을 세워 분석

import os
import subprocess
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import torch
import warnings
import threading
import signal
import sys
import faulthandler
import multiprocessing as mp
from itertools import islice
import gc

# 세그폴트 방지 설정
faulthandler.enable()

def segfault_handler(sig, frame):
    faulthandler.dump_traceback()
    print("Segmentation fault detected, cleaning up...")
    sys.exit(1)

signal.signal(signal.SIGSEGV, segfault_handler)

# 배치 처리 최적화 멀티프로세싱 설정
def setup_batch_optimized_multiprocessing():
    """배치 처리 최적화를 위한 설정"""
    try:
        mp.set_start_method('spawn', force=True)
        
        # 배치 처리 최적화 환경 변수
        os.environ.update({
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256",
            "CUDA_LAUNCH_BLOCKING": "0",
            "PYTHONWARNINGS": "ignore:semaphore_tracker:UserWarning"
        })
        
        print("배치 처리 최적화 설정 완료[2][4]")
        return True
    except Exception as e:
        print(f"멀티프로세싱 설정 실패: {e}")
        return False

setup_batch_optimized_multiprocessing()

# 메모리 최적화 강화
def setup_memory_optimization():
    try:
        gc.disable()
        gc.set_threshold(0)
        print("메모리 최적화 완료[2][3]")
        return True
    except Exception as e:
        print(f"메모리 최적화 실패: {e}")
        return False

setup_memory_optimization()

# CSV 저장 시 메모리 리셋 함수
def reset_memory_after_csv_save():
    try:
        # 1. 파이썬 가비지 컬렉션
        gc.enable()
        collected = gc.collect()
        gc.disable()
        
        # 2. GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
        
        # 3. NumPy 메모리 정리
        import numpy as np
        np.seterr(all='ignore')
        
        print(f"메모리 리셋 완료: {collected}개 객체 정리")
        return True
        
    except Exception as e:
        print(f"메모리 리셋 실패: {e}")
        return False

# CSV 전체 소수점 정밀도 저장 설정
def save_csv_with_full_precision(df, filepath):
    try:
        #  기본 pandas 저장으로 전체 정밀도 유지
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        print(f"전체 정밀도 CSV 저장: {filepath}")
        return True
    except Exception as e:
        print(f"CSV 저장 실패: {e}")
        return False

# CSV 전체 정밀도 읽기 함수
def read_csv_with_full_precision(filepath):
    try:
        # float_precision='round_trip'은 읽기에서만 사용[1]
        df = pd.read_csv(filepath, float_precision='round_trip')
        print(f"전체 정밀도 CSV 읽기: {filepath}")
        return df
    except Exception as e:
        print(f"CSV 읽기 실패: {e}")
        return pd.DataFrame()

# transformers 안전 임포트
try:
    from transformers import (
        Wav2Vec2ForSequenceClassification, 
        Wav2Vec2FeatureExtractor,
        AutoFeatureExtractor
    )
    TRANSFORMERS_AVAILABLE = True
    print("GPU 감정 분석 모델 임포트 성공")
except ImportError as e:
    print(f"Transformers 임포트 실패: {e}")
    TRANSFORMERS_AVAILABLE = False

warnings.filterwarnings('ignore')

# 효율적인 폴더 구조
SAVE_DIR = "./youtube/temp_audio"
TEMP_DIR = "./youtube/tempAudio"  
CPU_OUTPUT_FILE = "./youtube/cpu_audio_features.csv"
GPU_OUTPUT_FILE = "./youtube/gpu_emotion_analysis.csv"
FINAL_OUTPUT_FILE = "./youtube/final_merged_analysis.csv"
CACHE_DIR = "./youtube/cache"

# 단계별 배치 처리 설정 (사용자 요구사항)
CPU_BATCH_SIZE = 15
GPU_BATCH_SIZE = 5
SAVE_BATCH_SIZE = 15

for directory in [SAVE_DIR, TEMP_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# GPU 설정 최적화
def setup_batch_gpu():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.init()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.memory.set_per_process_memory_fraction(0.8)
            
            print(f"배치 처리 GPU 설정: {torch.cuda.get_device_name(0)}[6]")
        else:
            print("CUDA 사용 불가, CPU 모드")
        return device
    except Exception as e:
        print(f"GPU 설정 실패: {e}")
        return torch.device('cpu')

device = setup_batch_gpu()

class AccurateSevenEmotionMusicAnalyzer:    
    def __init__(self):
        self.models = {}
        self.is_loaded = False
        self.inference_lock = threading.Lock()
        
        # 정확한 7가지 감정 라벨링 (공식 정보 기반)
        self.emotion_mapping = {
            'happy': 0, 'sad': 1, 'angry': 2, 'fear': 3, 'surprise': 4, 'disgust': 5, 'neutral': 6
        }
        
        # 역매핑 (라벨 → 감정명)
        self.label_to_emotion = {v: k for k, v in self.emotion_mapping.items()}
        
        # 감정명 매핑 (한국어-영어)
        self.emotion_korean_mapping = {
            'happy': '기쁨', 'sad': '슬픔', 'angry': '분노', 
            'fear': '공포', 'surprise': '놀람', 'disgust': '혐오', 'neutral': '중립'
        }
        
        print(f"정확한 7가지 감정 분류기 초기화 (전체 정밀도):")
        print(f"감정: {list(self.emotion_mapping.keys())}")
    
    def initialize_emotion_models(self):
        try:
            if not TRANSFORMERS_AVAILABLE:
                print("Transformers 없음")
                return False
                
            print("7가지 감정 분류 GPU 모델 로딩...")
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            try:
                # CPU에서 먼저 로드 후 GPU로 이동
                wav2vec2_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False,
                    cache_dir=CACHE_DIR,
                    ignore_mismatched_sizes=True,
                    device_map=None,
                )
                
                # 프로세서 로딩
                try:
                    wav2vec2_processor = AutoFeatureExtractor.from_pretrained(
                        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                        cache_dir=CACHE_DIR
                    )
                except:
                    wav2vec2_processor = Wav2Vec2FeatureExtractor.from_pretrained(
                        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                        cache_dir=CACHE_DIR
                    )
                
                # 안전하게 GPU로 이동
                if torch.cuda.is_available():
                    wav2vec2_model = wav2vec2_model.to(device)
                
                self.models['wav2vec2'] = {
                    'model': wav2vec2_model,
                    'processor': wav2vec2_processor,
                    'emotions': ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'],
                    'enabled': True
                }
                self.models['wav2vec2']['model'].eval()
                
                # GPU 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                print("7가지 감정 분류 GPU 모델 로드 성공")
                self.is_loaded = True
                return True
                
            except Exception as model_error:
                print(f"모델 로딩 세부 오류: {model_error}")
                self.models['wav2vec2'] = {'enabled': False}
                return False
            
        except Exception as e:
            print(f"감정 모델 초기화 실패: {e}")
            self.models['wav2vec2'] = {'enabled': False}
            return False
    
    def enhanced_emotion_classification_with_musical_analysis(self, audio_features, gpu_emotion_result):
        try:
            # 오디오 특성 분석
            pitch_mean = audio_features.get('pitch_mean', 0) or 0
            energy_mean = audio_features.get('energy_mean', 0) or 0
            centroid_mean = audio_features.get('centroid_mean', 1000) or 1000
            bmp = audio_features.get('bmp', 120) or 120
            speech_rate = audio_features.get('speech_rate', 0) or 0
            initial_silence = audio_features.get('initial_silence', 0) or 0
            
            gpu_emotion = gpu_emotion_result.get('emotion_name', 'neutral')
            gpu_confidence = gpu_emotion_result.get('confidence', 0.5)
            
            print(f"음악적 특성 분석: 피치={pitch_mean}, 에너지={energy_mean}, BMP={bmp}")
            
            # 클래식/OST (슬픔)
            sadness_indicators = {
                'low_energy': energy_mean < 0.15,          # 낮은 에너지
                'minor_key_pitch': pitch_mean and 140 <= pitch_mean <= 155,  # 단조 음계 특성
                'slow_tempo': bmp < 70,                    # 느린 템포
                'instrumental': speech_rate < 10,          # 악기 중심
                'contemplative_centroid': 500 <= centroid_mean <= 650,  # 사색적인 음색
                'emotional_build': initial_silence > 0.1   # 감정적 시작
            }

            # 각 기준에 해당하는 것의 비율을 score로 정함
            sadness_score = sum(sadness_indicators.values()) / len(sadness_indicators)
            
            # 행복 감정 특화 분류
            happiness_indicators = {
                'high_energy': energy_mean > 0.4,          # 높은 에너지
                'major_key': pitch_mean and pitch_mean > 160,  # 장조 음계
                'upbeat_tempo': bmp > 100,                 # 빠른 템포
                'bright_timbre': centroid_mean > 2000,     # 밝은 음색
                'rhythmic': speech_rate > 15              # 리드미컬한 특성
            }
            
            happiness_score = sum(happiness_indicators.values()) / len(happiness_indicators)
            
            # 분노 감정 특화 분류
            anger_indicators = {
                'very_high_energy': energy_mean > 0.6,     # 매우 높은 에너지
                'aggressive_timbre': centroid_mean > 3000, # 공격적인 음색
                'fast_aggressive_tempo': bmp > 130,        # 빠르고 공격적인 템포
                'high_vocal_intensity': speech_rate > 20,  # 높은 보컬 강도
                'sudden_dynamics': initial_silence < 0.05  # 갑작스러운 시작
            }
            
            anger_score = sum(anger_indicators.values()) / len(anger_indicators)
            
            # 공포 감정 특화 분류
            fear_indicators = {
                'tense_energy': 0.3 <= energy_mean <= 0.6, # 긴장된 에너지
                'dissonant_pitch': pitch_mean and pitch_mean > 180,  # 불협화음
                'unstable_tempo': 80 <= bmp <= 140,        # 불안정한 템포
                'eerie_timbre': 1800 <= centroid_mean <= 2500,  # 으스스한 음색
                'minimal_vocal': speech_rate < 5           # 최소한의 보컬
            }
            
            fear_score = sum(fear_indicators.values()) / len(fear_indicators)
            
            # 놀람 감정 특화 분류
            surprise_indicators = {
                'sudden_energy': energy_mean > 0.5,        # 갑작스러운 에너지
                'sharp_timbre': centroid_mean > 2500,      # 날카로운 음색
                'varied_tempo': bmp and (bmp < 60 or bmp > 160),  # 극단적인 템포
                'dynamic_vocal': speech_rate > 25,         # 역동적인 보컬
                'abrupt_start': initial_silence < 0.02     # 갑작스러운 시작
            }
            
            surprise_score = sum(surprise_indicators.values()) / len(surprise_indicators)
            
            # 혐오 감정 특화 분류
            disgust_indicators = {
                'unpleasant_energy': 0.2 <= energy_mean <= 0.4,  # 불쾌한 에너지
                'harsh_timbre': centroid_mean > 3500,      # 거친 음색
                'irregular_tempo': bmp and (70 <= bmp <= 90),  # 불규칙한 템포
                'distorted_vocal': speech_rate > 30,       # 왜곡된 보컬
                'uncomfortable_start': 0.1 <= initial_silence <= 0.3  # 불편한 시작
            }
            
            disgust_score = sum(disgust_indicators.values()) / len(disgust_indicators)
            
            # 중립 감정 특화 분류
            neutral_indicators = {
                'moderate_energy': 0.15 <= energy_mean <= 0.35,  # 보통 에너지
                'balanced_timbre': 1000 <= centroid_mean <= 2000,  # 균형잡힌 음색
                'steady_tempo': 90 <= bmp <= 120,          # 안정된 템포
                'balanced_vocal': 5 <= speech_rate <= 15,  # 균형잡힌 보컬
                'normal_start': 0.05 <= initial_silence <= 0.2  # 일반적인 시작
            }
            
            neutral_score = sum(neutral_indicators.values()) / len(neutral_indicators)
            
            # 최종 감정 결정
            emotion_scores = {
                'sad': sadness_score,
                'happy': happiness_score,
                'angry': anger_score,
                'fear': fear_score,
                'surprise': surprise_score,
                'disgust': disgust_score,
                'neutral': neutral_score
            }
            
            # 음악적 특성 기반 최고 점수 감정
            best_musical_emotion = max(emotion_scores, key=emotion_scores.get)
            best_musical_score = emotion_scores[best_musical_emotion]
            
            print(f"음악적 분석 결과: {best_musical_emotion}({best_musical_score}), GPU: {gpu_emotion}({gpu_confidence})")
            
            # 최종 감정 결정 로직
            final_emotion = gpu_emotion
            final_confidence = gpu_confidence
            
            # 음악적 특성이 강하게 나타나는 경우 (70% 이상) 음악적 분석 우선
            if best_musical_score >= 0.7:
                final_emotion = best_musical_emotion
                final_confidence = min(0.95, best_musical_score + 0.1)
                print(f"음악적 특성 우선 적용: {final_emotion} (점수: {best_musical_score})")
            
            # 슬픔 감정 강화
            elif sadness_score >= 0.5 and energy_mean < 0.12:  # 매우 낮은 에너지 + 슬픔 특성
                final_emotion = 'sad'
                final_confidence = min(0.90, sadness_score + 0.2)
                print(f"슬픔 감정 특별 케이스 적용: {final_emotion} (점수: {sadness_score})")
            
            # GPU와 음악적 분석이 일치하는 경우 신뢰도 향상
            elif best_musical_emotion == gpu_emotion:
                final_confidence = min(0.95, (best_musical_score + gpu_confidence) / 2 + 0.1)
                print(f"GPU+음악 일치: {final_emotion} (신뢰도 향상: {final_confidence})")
            
            # 전체 정밀도 적용
            final_emotion_label = self.emotion_mapping.get(final_emotion, 6)
            
            return {
                'emotion_name': final_emotion,
                'emotion_label': final_emotion_label,
                'confidence': final_confidence,
                'musical_analysis': {
                    'best_emotion': best_musical_emotion,
                    'scores': emotion_scores
                }
            }
            
        except Exception as e:
            print(f"정확한 감정 분류 실패: {e}")
            return {
                'emotion_name': 'neutral',
                'emotion_label': 6,
                'confidence': 0.5,
                'musical_analysis': {'error': str(e)}
            }
    
    def cpu_extract_audio_features(self, y, sr):
        try:
            features = {}
            
            # 오디오 전처리 및 검증
            if len(y) < sr * 0.1:
                print(f"오디오 길이 부족: {len(y)/sr:.2f}초")
                return self._get_default_features()
            
            # 노이즈 제거 및 정규화
            y_filtered = self._preprocess_audio(y, sr)
            
            print(f"오디오 정보: 길이={len(y)/sr:.2f}초, 샘플링율={sr}Hz")
            
            def extract_pitch():
                try:
                    # YIN 알고리즘
                    f0_yin = librosa.yin(y_filtered, 
                                    fmin=librosa.note_to_hz('C2'), 
                                    fmax=librosa.note_to_hz('C7'), 
                                    sr=sr)
                    
                    if not np.all(np.isnan(f0_yin)):
                        valid_f0 = f0_yin[~np.isnan(f0_yin)]
                        if len(valid_f0) > len(f0_yin) * 0.1:
                            result = float(np.mean(valid_f0))
                            print(f"🎼 YIN 피치: {result:.2f}Hz")
                            return result
                    
                    # PYIN
                    print("YIN 실패, PYIN 시도")
                    f0_pyin, voiced_flag, voiced_probs = librosa.pyin(
                        y_filtered, 
                        fmin=librosa.note_to_hz('C2'), 
                        fmax=librosa.note_to_hz('C7'), 
                        sr=sr,
                        threshold=0.1
                    )
                    
                    if not np.all(np.isnan(f0_pyin)):
                        # 확신도 높은 피치만 사용
                        confident_pitch = f0_pyin[voiced_probs > 0.5]
                        if len(confident_pitch) > 0:
                            result = float(np.mean(confident_pitch))
                            print(f"PYIN 피치: {result:.2f}Hz")
                            return result
                    
                    # Piptrack
                    print("PYIN 실패, piptrack 시도")
                    pitches, magnitudes = librosa.piptrack(
                        y=y_filtered, sr=sr, 
                        fmin=librosa.note_to_hz('C2'), 
                        fmax=librosa.note_to_hz('C7'),
                        threshold=0.1
                    )
                    
                    pitch_values = []
                    for t in range(pitches.shape[1]):
                        index = magnitudes[:, t].argmax()
                        pitch = pitches[index, t]
                        if pitch > 0 and magnitudes[index, t] > 0.1:
                            pitch_values.append(pitch)
                    
                    if len(pitch_values) > 5:
                        result = float(np.median(pitch_values))
                        print(f"piptrack 피치: {result:.2f}Hz")
                        return result
                    
                    print("모든 피치 추출 방법 실패")
                    return None
                    
                except Exception as e:
                    print(f"피치 추출 오류: {type(e).__name__}: {e}")
                    return None
            
            def extract_energy():
                try:
                    # RMS 에너지
                    rms = librosa.feature.rms(y=y_filtered, frame_length=2048, hop_length=512)
                    rms_mean = float(np.mean(rms))
                    
                    # Zero Crossing Rate
                    zcr = librosa.feature.zero_crossing_rate(y_filtered)
                    zcr_mean = float(np.mean(zcr))
                    
                    # 조합 에너지 지표
                    combined_energy = rms_mean * (1 + zcr_mean)
                    
                    print(f"에너지: RMS={rms_mean:.4f}, ZCR={zcr_mean:.4f}, 조합={combined_energy:.4f}")
                    return combined_energy
                    
                except Exception as e:
                    print(f"에너지 추출 오류: {type(e).__name__}: {e}")
                    return 0.0
            
            def extract_centroid():
                try:
                    # 기본 스펙트럴 센트로이드
                    centroid = librosa.feature.spectral_centroid(y=y_filtered, sr=sr)
                    centroid_mean = float(np.mean(centroid))
                    
                    # 스펙트럴 롤오프 (보완 지표)
                    rolloff = librosa.feature.spectral_rolloff(y=y_filtered, sr=sr, roll_percent=0.85)
                    rolloff_mean = float(np.mean(rolloff))
                    
                    # 가중 평균 (센트로이드 70%, 롤오프 30%)
                    weighted_centroid = centroid_mean * 0.7 + rolloff_mean * 0.3
                    
                    print(f"센트로이드: 기본={centroid_mean:.1f}Hz, 롤오프={rolloff_mean:.1f}Hz, 가중={weighted_centroid:.1f}Hz")
                    return weighted_centroid
                    
                except Exception as e:
                    print(f"센트로이드 추출 오류: {type(e).__name__}: {e}")
                    return 1000.0
            
            def extract_bmp():
                try:
                    # 기본 beat tracking
                    tempo, beats = librosa.beat.beat_track(y=y_filtered, sr=sr, hop_length=512)
                    
                    print(f"기본 템포: {tempo:.1f} BPM")
                    
                    # 확장된 범위 (40-300 BPM)
                    if 40 <= tempo <= 300:
                        return float(tempo)
                    
                    # Onset detection 기반
                    print("템포 범위 초과, onset 기반 방법 시도")
                    onset_frames = librosa.onset.onset_detect(
                        y=y_filtered, sr=sr, units='time', 
                        backtrack=True, normalize=True
                    )
                    
                    if len(onset_frames) > 3:
                        intervals = np.diff(onset_frames)
                        median_interval = np.median(intervals)
                        valid_intervals = intervals[
                            (intervals > median_interval * 0.5) & 
                            (intervals < median_interval * 2.0)
                        ]
                        
                        if len(valid_intervals) > 0:
                            avg_interval = np.mean(valid_intervals)
                            onset_bpm = 60.0 / avg_interval if avg_interval > 0 else None
                            
                            if onset_bpm and 40 <= onset_bpm <= 300:
                                print(f"onset 템포: {onset_bpm:.1f} BPM")
                                return float(onset_bpm)
                    
                    # 템포그램 기반
                    print("onset 실패, 템포그램 시도")
                    tempogram = librosa.feature.tempogram(y=y_filtered, sr=sr)
                    tempo_freqs = librosa.tempo_frequencies(len(tempogram), sr=sr)
                    
                    # 가장 강한 템포 주파수 찾기
                    tempo_strength = np.mean(tempogram, axis=1)
                    max_tempo_idx = np.argmax(tempo_strength)
                    tempogram_bpm = tempo_freqs[max_tempo_idx] * 60
                    
                    if 40 <= tempogram_bpm <= 300:
                        print(f"템포그램 템포: {tempogram_bpm:.1f} BPM")
                        return float(tempogram_bpm)
                    
                    print("모든 템포 추출 방법 실패")
                    return None
                    
                except Exception as e:
                    print(f"BMP 추출 오류: {type(e).__name__}: {e}")
                    return None
            
            def extract_speech_rate():
                try:
                    # 단위 시간당 onset 수
                    onset_frames = librosa.onset.onset_detect(
                        y=y_filtered, sr=sr, units='time',
                        delta=0.05,
                        backtrack=True
                    )
                    
                    duration = len(y_filtered) / sr
                    speech_rate = len(onset_frames) / duration * 60
                    
                    print(f"Speech Rate: {speech_rate:.1f} onsets/min")
                    return int(speech_rate)
                    
                except Exception as e:
                    print(f"Speech Rate 추출 오류: {type(e).__name__}: {e}")
                    return 0
            
            def extract_initial_silence():
                try:
                    # 더 세밀한 에너지 분석
                    frame_length = 1024
                    hop_length = 256
                    
                    rms = librosa.feature.rms(
                        y=y_filtered, 
                        frame_length=frame_length, 
                        hop_length=hop_length
                    )
                    rms_vals = rms[0]
                    
                    # 동적 임계값 계산
                    overall_rms = np.mean(rms_vals)
                    threshold = max(0.01, overall_rms * 0.05)
                    
                    # 연속된 프레임에서 임계값 초과하는 지점 찾기
                    above_threshold = rms_vals > threshold
                    
                    # 최소 3프레임 연속으로 임계값 초과하는 지점
                    for i in range(len(above_threshold) - 2):
                        if np.all(above_threshold[i:i+3]):
                            silence_duration = librosa.frames_to_time(
                                i, sr=sr, hop_length=hop_length
                            )
                            print(f"초기 무음: {silence_duration:.3f}초")
                            return float(silence_duration)
                    
                    # 임계값을 초과하는 프레임이 없으면 전체 길이
                    duration = len(y_filtered) / sr
                    print(f"전체 무음: {duration:.3f}초")
                    return min(60.0, duration)
                    
                except Exception as e:
                    print(f"초기 무음 추출 오류: {type(e).__name__}: {e}")
                    return 0.0
            
            # 배치 처리 최적화 CPU 병렬 특성 추출
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = {
                    'pitch': executor.submit(extract_pitch),
                    'energy': executor.submit(extract_energy),
                    'centroid': executor.submit(extract_centroid),
                    'bmp': executor.submit(extract_bmp),
                    'speech': executor.submit(extract_speech_rate),
                    'silence': executor.submit(extract_initial_silence)
                }
                
                results = {}
                for key, future in futures.items():
                    try:
                        results[key] = future.result(timeout=10)
                    except Exception as e:
                        print(f"{key} 추출 타임아웃/오류: {e}")
                        default_values = {
                            'pitch': None, 'energy': 0.0, 'centroid': 1000.0,
                            'bmp': None, 'speech': 0, 'silence': 0.0
                        }
                        results[key] = default_values[key]
            
            return {
                'pitch_mean': results['pitch'],
                'energy_mean': results['energy'],
                'centroid_mean': results['centroid'],
                'bmp': results['bmp'],
                'speech_rate': results['speech'],
                'initial_silence': results['silence']
            }
            
        except Exception as e:
            print(f"CPU 특성 추출 실패: {type(e).__name__}: {e}")
            return self._get_default_features()

    def _preprocess_audio(self, y, sr):
        try:
            # 1. DC 성분 제거
            y_filtered = y - np.mean(y)
            
            # 2. 정규화
            if np.max(np.abs(y_filtered)) > 0:
                y_filtered = y_filtered / np.max(np.abs(y_filtered))
            
            # 3. 간단한 저역 통과 필터
            from scipy import signal
            nyquist = sr / 2
            cutoff = min(8000, nyquist * 0.8)
            b, a = signal.butter(3, cutoff / nyquist, btype='low')
            y_filtered = signal.filtfilt(b, a, y_filtered)
            
            return y_filtered
            
        except Exception as e:
            print(f"전처리 실패, 원본 사용: {e}")
            return y

    def _get_default_features(self):
        return {
            'pitch_mean': None, 
            'energy_mean': 0.0, 
            'centroid_mean': 1000.0,
            'bmp': None, 
            'speech_rate': 0, 
            'initial_silence': 0.0
        }
    
    def gpu_emotion_analysis(self, audio_path):
        if not self.models.get('wav2vec2', {}).get('enabled', False):
            return {'emotion_name': 'neutral', 'emotion_label': 6, 'confidence': 0.5}
        
        with self.inference_lock:
            try:
                model_info = self.models['wav2vec2']
                model = model_info['model']
                processor = model_info['processor']
                
                # 안전한 오디오 로드
                command = ["ffmpeg", "-y", "-i", audio_path, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"]
                process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True, timeout=8)
                y = np.frombuffer(process.stdout, np.float32)
                
                if len(y) == 0:
                    raise Exception("빈 오디오")
                
                # 배치 처리 최적화 길이
                target_length = 16000 * 60
                if len(y) > target_length:
                    y = y[:target_length]
                elif len(y) < target_length:
                    y = np.pad(y, (0, target_length - len(y)), mode='constant')
                
                # GPU 추론
                inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {key: value.to(device) for key, value in inputs.items()}
                
                with torch.no_grad():
                    try:
                        if torch.cuda.is_available():
                            with torch.amp.autocast('cuda'):
                                outputs = model(**inputs)
                        else:
                            outputs = model(**inputs)
                        
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        scores = predictions[0].cpu().numpy()
                        
                    except RuntimeError as e:
                        # GPU 오류 시 CPU fallback
                        print(f"GPU 오류, CPU로 fallback: {e}")
                        inputs = {key: value.cpu() for key, value in inputs.items()}
                        model_cpu = model.cpu()
                        outputs = model_cpu(**inputs)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        scores = predictions[0].numpy()
                        model.to(device)  # 모델 다시 GPU로
                
                # 7가지 감정 매핑
                emotions = model_info['emotions']
                emotion_mapping_local = {
                    'angry': 'angry', 'calm': 'neutral', 'disgust': 'disgust',
                    'fearful': 'fear', 'happy': 'happy', 'neutral': 'neutral',
                    'sad': 'sad', 'surprised': 'surprise'
                }
                
                emotion_scores = {}
                for i, emotion in enumerate(emotions):
                    mapped_emotion = emotion_mapping_local.get(emotion, 'neutral')
                    if mapped_emotion in emotion_scores:
                        emotion_scores[mapped_emotion] += float(scores[i])
                    else:
                        emotion_scores[mapped_emotion] = float(scores[i])
                
                best_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = emotion_scores[best_emotion]
                emotion_label = self.emotion_mapping.get(best_emotion, 6)
                
                return {
                    'emotion_name': best_emotion,
                    'emotion_label': emotion_label,
                    'confidence': confidence
                }
                
            except Exception as e:
                print(f"GPU 감정 분석 실패: {e}")
                return {'emotion_name': 'neutral', 'emotion_label': 6, 'confidence': 0.5}
            finally:
                # GPU 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

# 전역 분석기 인스턴스
music_emotion_analyzer = AccurateSevenEmotionMusicAnalyzer()

def extract_video_id(url):
    try:
        parsed = urlparse(url)
        return parse_qs(parsed.query).get("v", [None])[0]
    except:
        return None

# 인덱스 순서 기반 URL 로드 함수
def load_urls_in_index_order():
    try:
        input_file = './youtube/allYoutubeInfo_themeFiltered.csv'
        
        # 파일 존재 확인
        if not os.path.exists(input_file):
            print(f"입력 파일 없음: {input_file}")
            return []
        
        # 인덱스 기준으로 정렬하여 로드
        df = pd.read_csv(input_file)
        df_sorted = df.sort_index()
        
        print(f"원본 CSV 로드: {len(df_sorted):,}개 (인덱스 순서 정렬)")
        
        # URL과 인덱스를 함께 반환
        url_list = []
        for idx, row in df_sorted.iterrows():
            video_url = row.get('video_url')
            if pd.notna(video_url):
                video_id = extract_video_id(video_url)
                if video_id:
                    url_list.append({
                        'index': idx,
                        'url': video_url,
                        'video_id': video_id
                    })
        
        print(f"유효한 URL: {len(url_list):,}개 (인덱스 순서 유지)")
        return url_list
        
    except Exception as e:
        print(f"URL 로드 실패: {e}")
        return []

# 중복 처리 방지 강화 함수
def get_processed_video_ids(output_file):
    try:
        if os.path.exists(output_file):
            df = read_csv_with_full_precision(output_file)
            processed_ids = set(df['video_id'].dropna().astype(str))
            print(f"이미 처리된 ID: {len(processed_ids)}개 (from {output_file})")
            return processed_ids
        else:
            print(f"기존 결과 파일 없음: {output_file}")
            return set()
    except Exception as e:
        print(f"처리된 ID 확인 실패: {e}")
        return set()

def safe_download_with_cookies(url, video_id):
    """안전한 브라우저 쿠키 다운로드 (배치 최적화)[6]"""
    output_path = os.path.join(SAVE_DIR, f"{video_id}.m4a")
    
    strategies = [
        # {
        #     'name': 'ios_client',  # DRM 우회 최적화
        #     'command': [
        #         "yt-dlp", url,
        #         "--cookies-from-browser", "firefox",
        #         "--extractor-args", "youtube:player_client=ios",
        #         "--user-agent", "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)",
        #         "--sleep-interval", "3",
        #         "-f", "bestaudio[ext=m4a]",
        #         "--download-sections", "*0-60",
        #         "--socket-timeout", "30",
        #         "--retries", "1",
        #         "-o", f"{SAVE_DIR}/{video_id}.%(ext)s",
        #         "--quiet"
        #     ]
        # },
        # {
        #     'name': 'android_fallback',
        #     'command': [
        #         "yt-dlp", url,
        #         "--cookies-from-browser", "firefox",
        #         "--extractor-args", "youtube:player_client=android",
        #         "--user-agent", "Mozilla/5.0 (Linux; Android 11; Pixel 5)",
        #         "--sleep-interval", "3",
        #         "-f", "bestaudio[ext=m4a]",
        #         "--download-sections", "*0-60",
        #         "--socket-timeout", "30",
        #         "--retries", "1",
        #         "-o", f"{SAVE_DIR}/{video_id}.%(ext)s",
        #         "--quiet"
        #     ]
        # },
        {
            'name': 'chrome',
            'command': [
                "yt-dlp", url,
                "--cookies-from-browser", "chrome",
                "--user-agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                "-f", "bestaudio[ext=m4a]",
                "--download-sections", "*0-60",
                "--socket-timeout", "30",
                "--retries", "1",
                "-o", f"{SAVE_DIR}/{video_id}.%(ext)s",
                "--quiet"
            ]
        }
    ]
    
    for strategy in strategies:
        try:
            subprocess.run(strategy['command'], check=True, timeout=60)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                return True
        except:
            continue
    return False

def cpu_load_audio_ffmpeg(path, sr=22050):
    try:
        command = ["ffmpeg", "-y", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-threads", "8", "-"]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True, timeout=6)
        audio_data = np.frombuffer(process.stdout, np.float32)
        return audio_data.copy(), sr
    except Exception as e:
        raise Exception(f"CPU 오디오 로딩 실패: {e}")

# CPU 다운로드 + 기본 정보 추출
def process_cpu_audio_features_only(url_data):
    url = url_data['url']
    video_id = url_data['video_id']
    index = url_data['index']
    
    result = {
        "index": index,
        "video_id": video_id, "url": url, "pitch_mean": None, "energy_mean": None,
        "centroid_mean": None, "bmp": None, "speech_rate": None, "initial_silence": None,
        "error": None
    }
    
    output_path = os.path.join(SAVE_DIR, f"{video_id}.m4a")

    try:
        # 다운로드
        if not safe_download_with_cookies(url, video_id):
            raise Exception("다운로드 실패")

        # CPU: 오디오 특성 추출
        y, sr = cpu_load_audio_ffmpeg(output_path)
        audio_features = music_emotion_analyzer.cpu_extract_audio_features(y, sr)
        
        # CPU 특성을 결과에 저장
        for key, value in audio_features.items():
            if key in result:
                result[key] = value

        # 파일 삭제하지 않고 유지
        print(f"{video_id}(idx:{index}): CPU 특성 완료 (파일 유지) - 피치={result['pitch_mean']}, 에너지={result['energy_mean']}")

    except Exception as e:
        result["error"] = str(e)
        print(f"{video_id}(idx:{index}): CPU 처리 실패 - {str(e)}")
        # 에러 시에만 파일 삭제
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except:
            pass

    return result

# GPU 감정 분석 (기존 파일 사용)
def process_gpu_emotion_only(video_data):
    video_id = video_data['video_id']
    output_path = os.path.join(SAVE_DIR, f"{video_id}.m4a")
    
    result = {
        "video_id": video_id, "emotion_name": None, "emotion_label": None, "confidence": None, "error": None
    }
    
    try:
        # 파일 존재 확인
        if not os.path.exists(output_path):
            raise Exception("오디오 파일 없음")
        
        # 감정 분석
        gpu_emotion = music_emotion_analyzer.gpu_emotion_analysis(output_path)
        
        # 7가지 감정 분류 (음악적 특성 통합)
        audio_features = {
            'pitch_mean': video_data.get('pitch_mean'),
            'energy_mean': video_data.get('energy_mean'),
            'centroid_mean': video_data.get('centroid_mean'),
            'bmp': video_data.get('bmp'),
            'speech_rate': video_data.get('speech_rate'),
            'initial_silence': video_data.get('initial_silence')
        }
        
        final_emotion = music_emotion_analyzer.enhanced_emotion_classification_with_musical_analysis(
            audio_features, gpu_emotion
        )
        
        result["emotion_name"] = final_emotion['emotion_name']
        result["emotion_label"] = final_emotion['emotion_label']
        result["confidence"] = final_emotion['confidence']

        print(f"{video_id}: GPU 감정 완료 - {final_emotion['emotion_name']}({final_emotion['confidence']})")

    except Exception as e:
        result["error"] = str(e)
        print(f"{video_id}: GPU 처리 실패 - {str(e)}")
    finally:
        # GPU 처리 완료 후 파일 삭제
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
                print(f"{video_id}: 파일 삭제 완료")
        except:
            pass

    return result

# 배치 처리 유틸리티 함수
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def safe_cleanup():
    try:
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 가비지 컬렉션[2]
        gc.enable()
        gc.collect()
        gc.disable()
        
        print("안전한 리소스 정리 완료[2]")
    except Exception as e:
        print(f"리소스 정리 오류: {e}")

def safe_signal_handler(signum, frame):
    print(f"\n종료 신호 수신 ({signum}). 안전하게 정리 중...")
    safe_cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, safe_signal_handler)
signal.signal(signal.SIGTERM, safe_signal_handler)

def cpu_stage_main():
    print(f"CPU 배치 크기: {CPU_BATCH_SIZE}개씩")
    print("CSV 저장 시마다 메모리 리셋")
    
    # 인덱스 순서로 URL 로드
    url_list = load_urls_in_index_order()
    if not url_list:
        print("URL 로드 실패")
        return False
    
    # 처리된 ID 확인
    processed_ids = get_processed_video_ids(CPU_OUTPUT_FILE)
    
    # 인덱스 순서대로 미처리 URL 필터링
    unprocessed_urls = [url_data for url_data in url_list if url_data['video_id'] not in processed_ids]
    print(f"CPU 처리 대상: {len(unprocessed_urls):,}개 (인덱스 순서 유지)")

    if len(unprocessed_urls) == 0:
        print("CPU 처리 모든 완료")
        return True

    start = time.time()
    batch_results = []
    batch_count = 0

    # CPU 배치 15개씩 처리
    print(f"CPU 배치 {CPU_BATCH_SIZE}개씩 처리 시작...")
    try:
        with ThreadPoolExecutor(max_workers=CPU_BATCH_SIZE) as executor:
            url_batches = list(chunks(unprocessed_urls, CPU_BATCH_SIZE))
            
            for batch_idx, url_batch in enumerate(tqdm(url_batches, desc=f"CPU {CPU_BATCH_SIZE}개씩")):
                # 각 배치를 병렬 처리
                futures = [executor.submit(process_cpu_audio_features_only, url_data) for url_data in url_batch]
                
                # 배치 내 결과 수집
                batch_results_temp = []
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=180)
                        batch_results_temp.append(result)
                        
                    except Exception as e:
                        print(f"CPU 배치 처리 오류: {e}")
                
                batch_results.extend(batch_results_temp)
                
                # 전체 정밀도로 배치 크기 15개씩 저장 + 메모리 리셋
                if len(batch_results) >= SAVE_BATCH_SIZE:
                    temp_path = os.path.join(TEMP_DIR, f"cpu_features_{batch_count:04d}.csv")
                    
                    # 수정된 전체 정밀도로 저장
                    df_temp = pd.DataFrame(batch_results[:SAVE_BATCH_SIZE])
                    save_csv_with_full_precision(df_temp, temp_path)
                    
                    # CSV 저장 후 메모리 리셋
                    reset_memory_after_csv_save()
                    
                    batch_results = batch_results[SAVE_BATCH_SIZE:]
                    batch_count += 1
                    
                    print(f"CPU 배치 {batch_count} 전체 정밀도 저장 완료 + 메모리 리셋")
                
                print(f"CPU 배치 {batch_idx + 1}/{len(url_batches)} 완료 (처리: {len(batch_results_temp)}개)")

        # 마지막 배치도 전체 정밀도 저장 + 메모리 리셋
        if batch_results:
            temp_path = os.path.join(TEMP_DIR, f"cpu_features_{batch_count:04d}.csv")
            df_temp = pd.DataFrame(batch_results)
            save_csv_with_full_precision(df_temp, temp_path)
            
            # 마지막 배치도 메모리 리셋
            reset_memory_after_csv_save()
            print(f"CPU 마지막 배치 전체 정밀도 저장 완료 + 메모리 리셋")

    except Exception as e:
        print(f"CPU 처리 오류: {e}")
        return False
    finally:
        safe_cleanup()

    elapsed_time = time.time() - start
    print(f"CPU 처리 시간: {elapsed_time:.2f}초")
    
    # CPU 결과 병합 + 메모리 리셋
    merge_cpu_batches()
    return True

def gpu_stage_main():
    print("2단계: GPU 감정 분석 시작")
    print(f"GPU 배치 크기: {GPU_BATCH_SIZE}개씩")
    print("CSV 저장 시마다 메모리 리셋[2]")
    
    # CPU 결과 파일 확인
    if not os.path.exists(CPU_OUTPUT_FILE):
        print(f"CPU 결과 파일 없음: {CPU_OUTPUT_FILE}")
        return False
    
    # GPU 모델 초기화
    if not music_emotion_analyzer.initialize_emotion_models():
        print("GPU 감정 모델 초기화 실패")
        return False
    
    # CPU 결과를 전체 정밀도로 인덱스 순서로 로드
    cpu_df = read_csv_with_full_precision(CPU_OUTPUT_FILE)
    cpu_df = cpu_df[cpu_df['error'].isna()]
    
    # 인덱스 기준으로 정렬
    if 'index' in cpu_df.columns:
        cpu_df = cpu_df.sort_values('index')
        print(f"GPU 처리 대상: {len(cpu_df):,}개 (인덱스 순서 정렬)")
    else:
        print(f"GPU 처리 대상: {len(cpu_df):,}개 (인덱스 정보 없음)")

    # 처리된 ID 확인
    processed_ids = get_processed_video_ids(GPU_OUTPUT_FILE)
    
    unprocessed_df = cpu_df[~cpu_df['video_id'].isin(processed_ids)]
    print(f"GPU 미처리 대상: {len(unprocessed_df):,}개")

    if len(unprocessed_df) == 0:
        print("GPU 처리 모든 완료")
        return True

    start = time.time()
    batch_results = []
    batch_count = 0

    # GPU 배치 5개씩 처리
    print(f"GPU 배치 {GPU_BATCH_SIZE}개씩 처리 시작...")
    try:
        # GPU는 순차 처리
        video_batches = list(chunks(unprocessed_df.to_dict('records'), GPU_BATCH_SIZE))
        
        for batch_idx, video_batch in enumerate(tqdm(video_batches, desc=f"🎮 GPU {GPU_BATCH_SIZE}개씩")):
            batch_results_temp = []
            
            for video_data in video_batch:
                try:
                    result = process_gpu_emotion_only(video_data)
                    batch_results_temp.append(result)
                    
                except Exception as e:
                    print(f"GPU 처리 오류: {e}")
                    batch_results_temp.append({
                        "video_id": video_data['video_id'], 
                        "emotion_name": None, "emotion_label": None, 
                        "confidence": None, "error": str(e)
                    })
            
            batch_results.extend(batch_results_temp)
            
            # 전체 정밀도로 배치 크기 5개씩 저장 + 메모리 리셋
            if len(batch_results) >= GPU_BATCH_SIZE:
                temp_path = os.path.join(TEMP_DIR, f"gpu_emotions_{batch_count:04d}.csv")
                
                # 전체 정밀도로 저장
                df_temp = pd.DataFrame(batch_results[:GPU_BATCH_SIZE])
                save_csv_with_full_precision(df_temp, temp_path)
                
                # CSV 저장 후 메모리 리셋
                reset_memory_after_csv_save()
                
                batch_results = batch_results[GPU_BATCH_SIZE:]
                batch_count += 1
                
                print(f"GPU 배치 {batch_count} 전체 정밀도 저장 완료 + 메모리 리셋")
            
            print(f"GPU 배치 {batch_idx + 1}/{len(video_batches)} 완료 (처리: {len(batch_results_temp)}개)")

        # 마지막 배치도 전체 정밀도 저장 + 메모리 리셋
        if batch_results:
            temp_path = os.path.join(TEMP_DIR, f"gpu_emotions_{batch_count:04d}.csv")
            df_temp = pd.DataFrame(batch_results)
            save_csv_with_full_precision(df_temp, temp_path)
            
            # 마지막 배치도 메모리 리셋
            reset_memory_after_csv_save()
            print(f"GPU 마지막 배치 전체 정밀도 저장 완료 + 메모리 리셋")

    except Exception as e:
        print(f"GPU 처리 오류: {e}")
        return False
    finally:
        safe_cleanup()

    elapsed_time = time.time() - start
    print(f"GPU 처리 시간: {elapsed_time:.2f}초")
    
    # GPU 결과 병합 + 메모리 리셋
    merge_gpu_batches()
    return True

def merge_cpu_batches():
    import glob
    
    temp_files = sorted(glob.glob(os.path.join(TEMP_DIR, "cpu_features_*.csv")))
    if not temp_files:
        print("CPU 병합할 파일 없음")
        return

    try:
        df_list = []
        for f in temp_files:
            df_temp = read_csv_with_full_precision(f)
            df_list.append(df_temp)
        
        df_all = pd.concat(df_list, ignore_index=True)
        
        # CPU 컬럼 순서
        cpu_columns = [
            'index', 'video_id', 'url', 'pitch_mean', 'energy_mean', 'centroid_mean', 
            'bmp', 'speech_rate', 'initial_silence', 'error'
        ]
        df_all = df_all.reindex(columns=[col for col in cpu_columns if col in df_all.columns])
        
        # 인덱스 순서로 정렬
        if 'index' in df_all.columns:
            df_all = df_all.sort_values('index')
            print("CPU 결과를 인덱스 순서로 정렬")
        
        # 🔢 수정된 전체 정밀도로 최종 저장[1][3]
        save_csv_with_full_precision(df_all, CPU_OUTPUT_FILE)
        print(f"CPU 결과 전체 정밀도 병합 완료: {len(df_all):,}개 레코드")
        
        # 병합 완료 후 메모리 리셋
        reset_memory_after_csv_save()
        
        # CPU 처리 통계
        success_count = len(df_all[df_all['error'].isna()])
        error_count = len(df_all[df_all['error'].notna()])
        print(f"CPU 처리 통계: 성공 {success_count}개, 실패 {error_count}개")
                
    except Exception as e:
        print(f"CPU 병합 오류: {e}")

def merge_gpu_batches():
    import glob
    
    temp_files = sorted(glob.glob(os.path.join(TEMP_DIR, "gpu_emotions_*.csv")))
    if not temp_files:
        print("GPU 병합할 파일 없음")
        return

    try:
        # 전체 정밀도 유지하여 로드
        df_list = []
        for f in temp_files:
            df_temp = read_csv_with_full_precision(f)
            df_list.append(df_temp)
        
        df_all = pd.concat(df_list, ignore_index=True)
        
        # GPU 컬럼 순서
        gpu_columns = [
            'video_id', 'emotion_name', 'emotion_label', 'confidence', 'error'
        ]
        df_all = df_all.reindex(columns=gpu_columns)
        
        # 수정된 전체 정밀도로 저장[1][3]
        save_csv_with_full_precision(df_all, GPU_OUTPUT_FILE)
        print(f"GPU 결과 전체 정밀도 병합 완료: {len(df_all):,}개 레코드")
        
        # 병합 완료 후 메모리 리셋
        reset_memory_after_csv_save()
        
        # GPU 처리 통계
        success_count = len(df_all[df_all['error'].isna()])
        error_count = len(df_all[df_all['error'].notna()])
        print(f"GPU 처리 통계: 성공 {success_count}개, 실패 {error_count}개")
        
        # 감정 분포
        if 'emotion_name' in df_all.columns:
            emotion_counts = df_all[df_all['error'].isna()]['emotion_name'].value_counts()
            print("감정 분포:")
            for emotion, count in emotion_counts.items():
                percentage = (count / success_count) * 100 if success_count > 0 else 0
                print(f"   {emotion}: {count}개 ({percentage:.1f}%)")
                
    except Exception as e:
        print(f"GPU 병합 오류: {e}")

def final_merge_csv_files():
    print("3단계: 최종 CSV 병합 (video_id 기준) + 전체 정밀도 + 메모리 리셋")
    
    # 파일 존재 확인
    if not os.path.exists(CPU_OUTPUT_FILE):
        print(f"CPU 결과 파일 없음: {CPU_OUTPUT_FILE}")
        return False
    
    if not os.path.exists(GPU_OUTPUT_FILE):
        print(f"GPU 결과 파일 없음: {GPU_OUTPUT_FILE}")
        return False
    
    try:
        # 전체 정밀도 유지하여 CSV 파일 로드
        cpu_df = read_csv_with_full_precision(CPU_OUTPUT_FILE)
        gpu_df = read_csv_with_full_precision(GPU_OUTPUT_FILE)
        
        print(f"CPU 데이터: {len(cpu_df):,}개")
        print(f"GPU 데이터: {len(gpu_df):,}개")
        
        # video_id 기준으로 병합
        merged_df = pd.merge(
            cpu_df[cpu_df['error'].isna()],
            gpu_df[gpu_df['error'].isna()],
            on='video_id', 
            how='inner'
        )
        
        # 인덱스 순서로 정렬
        if 'index' in merged_df.columns:
            merged_df = merged_df.sort_values('index')
            print("최종 결과를 인덱스 순서로 정렬")
        
        # 최종 컬럼 순서
        final_columns = [
            'video_id', 'pitch_mean', 'energy_mean', 'centroid_mean', 
            'bmp', 'speech_rate', 'initial_silence', 
            'emotion_name', 'emotion_label', 'confidence'
        ]
        merged_df = merged_df.reindex(columns=final_columns)
        
        # 최종 전체 정밀도로 파일 저장
        save_csv_with_full_precision(merged_df, FINAL_OUTPUT_FILE)
        
        # 최종 저장 후 메모리 리셋
        reset_memory_after_csv_save()
        
        print(f"최종 전체 정밀도 병합 완료: {len(merged_df):,}개 레코드 → {FINAL_OUTPUT_FILE}[3]")
        
        # 최종 통계
        print(f"성공적으로 병합된 레코드: {len(merged_df):,}개")
        print(f"평균 신뢰도: {merged_df['confidence'].mean()}")
        
        # 감정 분포 최종
        emotion_counts = merged_df['emotion_name'].value_counts()
        print("최종 감정 분포:")
        for emotion, count in emotion_counts.items():
            percentage = (count / len(merged_df)) * 100
            print(f"   {emotion}: {count}개 ({percentage:.1f}%)")
        
        return True
                
    except Exception as e:
        print(f"최종 병합 오류: {e}")
        return False

def main():
    print("1단계: CPU 다운로드 + 기본 정보 (배치 15개)")
    print("2단계: GPU 감정 분석 (배치 5개)")
    print("3단계: 최종 병합")
    print("모든 CSV 저장 시 메모리 리셋 적용")
    print("전체 소수점 정밀도 저장")
    
    # CPU 처리
    if not cpu_stage_main():
        print("1단계 CPU 처리 실패")
        return
    
    # GPU 처리
    if not gpu_stage_main():
        print("2단계 GPU 처리 실패")
        return
    
    # 최종 병합
    if not final_merge_csv_files():
        print("3단계 최종 병합 실패")
        return
    
    print(f"최종 결과: {FINAL_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
