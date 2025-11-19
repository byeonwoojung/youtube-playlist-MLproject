"""
공통 유틸리티 함수 모듈
GPU 설정, CSV 저장/로드 등의 공통 기능 제공
"""

import os
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
import gc
from typing import Optional


def setup_gpu():
    """
    GPU 설정 및 최적화
    TensorFlow와 PyTorch GPU 환경을 자동으로 설정
    
    Returns:
        dict: GPU 설정 정보
    """
    gpu_info = {
        "tensorflow_gpu": False,
        "pytorch_cuda": False,
        "device": "cpu"
    }
    
    # TensorFlow GPU 설정
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ TensorFlow GPU 활성화: {len(gpus)}개")
            gpu_info["tensorflow_gpu"] = True
        except RuntimeError as e:
            print(f"⚠ TensorFlow GPU 설정 오류: {e}")
    
    # PyTorch CUDA 설정
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"✓ PyTorch CUDA 활성화: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        gpu_info["pytorch_cuda"] = True
        gpu_info["device"] = "cuda:0"
    else:
        print("⚠ CUDA 사용 불가, CPU 사용")
        gpu_info["device"] = "cpu"
    
    return gpu_info


def save_csv_safely(df: pd.DataFrame, filepath: str, encoding: str = "utf-8-sig") -> bool:
    """
    전체 정밀도를 유지하면서 CSV 파일로 안전하게 저장
    
    Args:
        df: 저장할 DataFrame
        filepath: 저장 경로
        encoding: 인코딩 방식 (기본값: utf-8-sig)
    
    Returns:
        bool: 저장 성공 여부
    """
    try:
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 전체 정밀도 유지하며 저장
        df.to_csv(filepath, index=False, encoding=encoding)
        print(f"✅ CSV 저장 완료: {filepath}")
        
        # 메모리 정리
        reset_memory()
        return True
    except Exception as e:
        print(f"❌ CSV 저장 실패 ({filepath}): {e}")
        return False


def load_csv_safely(filepath: str, encoding: str = "utf-8") -> Optional[pd.DataFrame]:
    """
    전체 정밀도를 유지하면서 CSV 파일 로드
    
    Args:
        filepath: 로드할 파일 경로
        encoding: 인코딩 방식 (기본값: utf-8)
    
    Returns:
        DataFrame 또는 None (실패 시)
    """
    try:
        # float_precision='round_trip'으로 정밀도 유지
        df = pd.read_csv(filepath, encoding=encoding, float_precision='round_trip')
        print(f"✅ CSV 로드 완료: {filepath} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"❌ CSV 로드 실패 ({filepath}): {e}")
        return None


def reset_memory():
    """
    메모리 정리 (Python GC + GPU 캐시)
    CSV 저장 후 또는 대용량 작업 후 호출
    """
    try:
        # Python 가비지 컬렉션
        gc.enable()
        collected = gc.collect()
        gc.disable()
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
        
        print(f"🧹 메모리 정리 완료: {collected}개 객체 제거")
        return True
    except Exception as e:
        print(f"⚠ 메모리 정리 실패: {e}")
        return False


def extract_video_id(url: str) -> Optional[str]:
    """
    YouTube URL에서 video_id 추출
    
    Args:
        url: YouTube 영상 URL
    
    Returns:
        video_id (11자리) 또는 None
    """
    from urllib.parse import urlparse, parse_qs
    
    try:
        parsed = urlparse(url)
        video_id = parse_qs(parsed.query).get("v", [None])[0]
        return video_id
    except Exception:
        return None


def setup_multiprocessing_optimized():
    """
    멀티프로세싱 최적화 설정
    배치 처리 시 호출
    """
    import multiprocessing as mp
    
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
        
        print("✓ 멀티프로세싱 최적화 설정 완료")
        return True
    except Exception as e:
        print(f"⚠ 멀티프로세싱 설정 실패: {e}")
        return False
