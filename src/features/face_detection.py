import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from tqdm import tqdm
import os
import time
from typing import List, Tuple, Optional, Dict
import json
from pathlib import Path
import gc

class GPUFaceAnalyzer:
    def __init__(self, model_path: str = 'face_detection_yunet_2023mar.onnx'):
        """GPU 가속 얼굴 분석기 초기화"""
        self.setup_gpu()
        self.detector = self._load_yunet_model(model_path)
        self.device = self._get_optimal_device()
        
    def setup_gpu(self):
        """GPU 설정 및 최적화"""
        # TensorFlow GPU 설정
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✓ TensorFlow GPU 활성화: {len(gpus)}개")
            except RuntimeError as e:
                print(f"⚠ TensorFlow GPU 설정 오류: {e}")
        
        # PyTorch CUDA 설정
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            print(f"✓ PyTorch CUDA 활성화: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            print("⚠ CUDA 사용 불가, CPU 사용")
    
    def _get_optimal_device(self):
        """최적 디바이스 선택"""
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    
    def _load_yunet_model(self, model_path: str):
        """YuNet 모델 로드"""
        if not os.path.exists(model_path):
            print(f"⚠ 모델 파일을 찾을 수 없습니다: {model_path}")
            print("다음 링크에서 모델을 다운로드하세요:")
            print("https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet")
            return None
        
        try:
            detector = cv2.FaceDetectorYN_create(
                model_path, "", (320, 320), 0.8, 0.3, 5000
            )
            print(f"✓ YuNet 모델 로드 완료: {model_path}")
            return detector
        except Exception as e:
            print(f"✗ 모델 로드 실패: {e}")
            return None

    def detect_faces_optimized(self, image: np.ndarray) -> List[np.ndarray]:
        """최적화된 얼굴 탐지"""
        if self.detector is None:
            return []
        
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        
        try:
            _, faces = self.detector.detect(image)
            if faces is None:
                return []
            return faces
        except Exception as e:
            print(f"⚠ 얼굴 탐지 오류: {e}")
            return []
    
    def calculate_face_ratio_vectorized(self, faces: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """벡터화된 얼굴 비율 계산"""
        if len(faces) == 0:
            return np.array([])
        
        face_areas = faces[:, 2] * faces[:, 3]
        image_area = image_shape[0] * image_shape[1]
        return (face_areas / image_area) * 100
    
    def is_frontal_face_advanced(self, faces: np.ndarray) -> np.ndarray:
        """Roll은 관대하고 Yaw는 엄격한 정면 얼굴 판단"""
        if len(faces) == 0:
            return np.array([], dtype=bool)
        
        frontal_flags = np.zeros(len(faces), dtype=bool)
        
        for i, face in enumerate(faces):
            if len(face) < 14:
                continue
            
            try:
                # 랜드마크 추출
                left_eye = np.array([face[4], face[5]])
                right_eye = np.array([face[6], face[7]])
                nose_tip = np.array([face[8], face[9]])
                left_mouth = np.array([face[10], face[11]])
                right_mouth = np.array([face[12], face[13]])
                
                # 점수 기반 평가 시스템
                frontal_score = 0
                
                # 1. 코 중심 대칭성 (Yaw 판단) - 더욱 엄격하게
                nose_x = nose_tip[0]
                left_eye_dist = abs(left_eye[0] - nose_x)
                right_eye_dist = abs(right_eye[0] - nose_x)
                left_mouth_dist = abs(left_mouth[0] - nose_x)
                right_mouth_dist = abs(right_mouth[0] - nose_x)
                
                # 눈 대칭성 - 엄격한 기준
                if right_eye_dist > 0 and left_eye_dist > 0:
                    eye_symmetry = min(left_eye_dist, right_eye_dist) / max(left_eye_dist, right_eye_dist)
                    if eye_symmetry >= 0.85:  # 0.75 → 0.85로 다시 엄격하게
                        frontal_score += 2  # 가중치 증가
                    elif eye_symmetry >= 0.75:
                        frontal_score += 1  # 부분 점수
                
                # 입 대칭성 - 엄격한 기준
                if right_mouth_dist > 0 and left_mouth_dist > 0:
                    mouth_symmetry = min(left_mouth_dist, right_mouth_dist) / max(left_mouth_dist, right_mouth_dist)
                    if mouth_symmetry >= 0.85:  # 0.75 → 0.85로 다시 엄격하게
                        frontal_score += 2  # 가중치 증가
                    elif mouth_symmetry >= 0.75:
                        frontal_score += 1  # 부분 점수
                
                # 2. 수직 정렬도 (Yaw 판단) - 더욱 엄격하게
                face_width = abs(right_eye[0] - left_eye[0])
                if face_width > 0:
                    eye_center_x = (left_eye[0] + right_eye[0]) / 2
                    mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
                    
                    # 코와 중심선의 정렬도
                    nose_to_eye_center = abs(nose_x - eye_center_x) / face_width
                    nose_to_mouth_center = abs(nose_x - mouth_center_x) / face_width
                    
                    # 엄격한 정렬 기준
                    if nose_to_eye_center <= 0.08 and nose_to_mouth_center <= 0.08:  # 0.15 → 0.08로 엄격하게
                        frontal_score += 2
                    elif nose_to_eye_center <= 0.12 and nose_to_mouth_center <= 0.12:
                        frontal_score += 1
                
                # 3. 추가 Yaw 검증 - 얼굴 특징점 비율 확인
                if face_width > 0:
                    # 좌우 눈과 코의 거리 비율 (더 엄격한 Yaw 판단)
                    left_nose_ratio = left_eye_dist / face_width
                    right_nose_ratio = right_eye_dist / face_width
                    
                    # 좌우 비율이 너무 차이나면 Yaw 회전으로 판단
                    ratio_diff = abs(left_nose_ratio - right_nose_ratio)
                    if ratio_diff <= 0.10:  # 10% 이하 차이만 허용
                        frontal_score += 2
                    elif ratio_diff <= 0.15:  # 15% 이하는 부분 점수
                        frontal_score += 1
                
                # 4. 눈과 입의 수평도 (Roll 판단) - 관대하게 유지
                if face_width > 0:
                    eye_slope = abs(left_eye[1] - right_eye[1]) / face_width
                    mouth_slope = abs(left_mouth[1] - right_mouth[1]) / face_width
                    
                    # Roll 각도에 관대한 기준 (60도까지 허용)
                    if eye_slope <= 0.87 and mouth_slope <= 0.87:  # sin(60°) ≈ 0.87
                        frontal_score += 1
                    
                    # 눈과 입이 비슷한 기울기면 추가 점수
                    slope_diff = abs(eye_slope - mouth_slope)
                    if slope_diff <= 0.3:
                        frontal_score += 0.5
                
                # 5. Roll 각도 보정 (관대하게 유지)
                if face_width > 0:
                    roll_angle = np.arctan(abs(left_eye[1] - right_eye[1]) / face_width) * 180 / np.pi
                    
                    if roll_angle <= 60:  # 60도까지 허용
                        frontal_score += 1
                    elif roll_angle <= 70:
                        frontal_score += 0.5
                
                # 6. 전체적인 얼굴 비율 - 적당히 유지
                eye_width = abs(right_eye[0] - left_eye[0])
                mouth_width = abs(right_mouth[0] - left_mouth[0])
                
                if eye_width > 0 and mouth_width > 0:
                    width_ratio = min(eye_width, mouth_width) / max(eye_width, mouth_width)
                    if width_ratio >= 0.60:
                        frontal_score += 1
                
                # 최종 판단: 11점 중 7점 이상이면 정면 (Yaw에 엄격, Roll에 관대)
                frontal_flags[i] = frontal_score >= 7.0
                
                # 디버깅 출력
                if frontal_score >= 5.0:
                    roll_angle_debug = np.arctan(abs(left_eye[1] - right_eye[1]) / face_width) * 180 / np.pi if face_width > 0 else 0
                    yaw_indicator = f"대칭:{eye_symmetry:.2f}/{mouth_symmetry:.2f}" if 'eye_symmetry' in locals() and 'mouth_symmetry' in locals() else "N/A"
                    print(f"   얼굴 #{i+1}: {frontal_score:.1f}/11점, Roll:{roll_angle_debug:.1f}°, Yaw:{yaw_indicator} {'✅정면' if frontal_flags[i] else '❌비정면'}")
                    
            except Exception as e:
                frontal_flags[i] = False
                continue
        
        return frontal_flags


    
    # def is_frontal_face_advanced(self, faces: np.ndarray) -> np.ndarray:
    #     """60도 Roll 각도까지 허용하는 정면 얼굴 판단"""
    #     if len(faces) == 0:
    #         return np.array([], dtype=bool)
        
    #     frontal_flags = np.zeros(len(faces), dtype=bool)
        
    #     for i, face in enumerate(faces):
    #         if len(face) < 14:
    #             continue
            
    #         try:
    #             # 랜드마크 추출
    #             left_eye = np.array([face[4], face[5]])
    #             right_eye = np.array([face[6], face[7]])
    #             nose_tip = np.array([face[8], face[9]])
    #             left_mouth = np.array([face[10], face[11]])
    #             right_mouth = np.array([face[12], face[13]])
                
    #             # 점수 기반 평가 시스템
    #             frontal_score = 0
                
    #             # 1. 코 중심 대칭성 (Yaw 판단) - 엄격 유지
    #             nose_x = nose_tip[0]
    #             left_eye_dist = abs(left_eye[0] - nose_x)
    #             right_eye_dist = abs(right_eye[0] - nose_x)
    #             left_mouth_dist = abs(left_mouth[0] - nose_x)
    #             right_mouth_dist = abs(right_mouth[0] - nose_x)
                
    #             if right_eye_dist > 0 and left_eye_dist > 0:
    #                 eye_symmetry = min(left_eye_dist, right_eye_dist) / max(left_eye_dist, right_eye_dist)
    #                 if eye_symmetry >= 0.75:
    #                     frontal_score += 1
                
    #             if right_mouth_dist > 0 and left_mouth_dist > 0:
    #                 mouth_symmetry = min(left_mouth_dist, right_mouth_dist) / max(left_mouth_dist, right_mouth_dist)
    #                 if mouth_symmetry >= 0.75:
    #                     frontal_score += 1
                
    #             # 2. 수직 정렬도 (Yaw 판단) - 엄격 유지
    #             face_width = abs(right_eye[0] - left_eye[0])
    #             if face_width > 0:
    #                 eye_center_x = (left_eye[0] + right_eye[0]) / 2
    #                 mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
                    
    #                 nose_alignment = min(
    #                     abs(nose_x - eye_center_x) / face_width,
    #                     abs(nose_x - mouth_center_x) / face_width
    #                 )
                    
    #                 if nose_alignment <= 0.15:
    #                     frontal_score += 1
                
    #             # 3. 눈과 입의 수평도 (Roll 판단) - 60도까지 대폭 완화
    #             if face_width > 0:
    #                 eye_slope = abs(left_eye[1] - right_eye[1]) / face_width
    #                 mouth_slope = abs(left_mouth[1] - right_mouth[1]) / face_width
                    
    #                 # 60도 기울기까지 허용 (sin(60°) ≈ 0.87)
    #                 if eye_slope <= 0.87 and mouth_slope <= 0.87:  # 0.4 → 0.87로 대폭 완화
    #                     frontal_score += 1
    #                 # 추가 점수: 눈과 입이 비슷한 기울기면 추가 점수
    #                 slope_diff = abs(eye_slope - mouth_slope)
    #                 if slope_diff <= 0.3:  # 0.2 → 0.3으로 완화
    #                     frontal_score += 0.5
                
    #             # 4. 전체적인 얼굴 비율 - 더 완화
    #             eye_width = abs(right_eye[0] - left_eye[0])
    #             mouth_width = abs(right_mouth[0] - left_mouth[0])
                
    #             if eye_width > 0 and mouth_width > 0:
    #                 width_ratio = min(eye_width, mouth_width) / max(eye_width, mouth_width)
    #                 if width_ratio >= 0.50:  # 0.60 → 0.50으로 더 완화
    #                     frontal_score += 1
                
    #             # 5. Roll 각도 보정 추가 점수 (60도까지 허용)
    #             if face_width > 0:
    #                 # 눈의 기울기를 Roll 각도로 변환 (근사치)
    #                 roll_angle = np.arctan(abs(left_eye[1] - right_eye[1]) / face_width) * 180 / np.pi
                    
    #                 # 60도 이하의 Roll은 정면으로 간주
    #                 if roll_angle <= 60:  # 40도 → 60도로 확대
    #                     frontal_score += 1
    #                 elif roll_angle <= 70:  # 60-70도는 부분 점수
    #                     frontal_score += 0.5
    #                 elif roll_angle <= 80:  # 70-80도는 소량 점수
    #                     frontal_score += 0.3
                
    #             # 6. 추가 관대한 기준 (60도 허용을 위한)
    #             if face_width > 0:
    #                 # 극단적 기울기에도 대응
    #                 max_slope = max(eye_slope, mouth_slope) if 'eye_slope' in locals() and 'mouth_slope' in locals() else 0
    #                 if max_slope <= 1.0:  # tan(45°) = 1.0, 더 관대하게
    #                     frontal_score += 0.5
                
    #             # 최종 판단: 7점 중 4점 이상이면 정면 (더 관대한 기준)
    #             frontal_flags[i] = frontal_score >= 4.0
                
    #             # 디버깅 출력
    #             if frontal_score >= 3.5:  # 거의 정면인 경우 로그 출력
    #                 roll_angle_debug = np.arctan(abs(left_eye[1] - right_eye[1]) / face_width) * 180 / np.pi if face_width > 0 else 0
    #                 print(f"   얼굴 #{i+1}: 점수 {frontal_score:.1f}/7, Roll각도 {roll_angle_debug:.1f}° {'✅정면' if frontal_flags[i] else '❌비정면'}")
                    
    #         except Exception as e:
    #             frontal_flags[i] = False
    #             continue
        
    #     return frontal_flags



    def process_single_image(self, image_path: str) -> Dict:
        """단일 이미지 처리 (8% 정면 + 8% 분리)"""
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"이미지 로드 실패: {image_path}"}
        
        h, w = image.shape[:2]
        faces = self.detect_faces_optimized(image)
        
        if len(faces) == 0:
            return {
                "image_path": str(image_path),
                "group_id": Path(image_path).name[:11],
                "image_count": 1,
                "image_size": [int(h), int(w)],
                "total_faces": 0,
                "all_faces_data": [],
                "faces_8_percent": [],
                "frontal_faces_8_percent": []
            }
        
        # 모든 얼굴의 비율 계산
        face_ratios = self.calculate_face_ratio_vectorized(faces, (h, w))
        
        # 모든 얼굴 데이터 저장
        all_faces_data = []
        for i, (face, ratio) in enumerate(zip(faces, face_ratios)):
            x, y, width, height = face[:4].astype(int)
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            width = max(1, min(width, w-x))
            height = max(1, min(height, h-y))
            
            face_data = {
                "face_id": int(i),
                "bbox": [int(x), int(y), int(width), int(height)],
                "confidence": float(face[14]) if len(face) > 14 else 0.0,
                "face_ratio_percent": float(round(ratio, 2)),
                "landmarks": [int(coord) for coord in face[4:14]] if len(face) >= 14 else []
            }
            all_faces_data.append(face_data)
        
        # 8% 이상 얼굴 (기존 유지)
        large_face_mask = face_ratios >= 8.0
        large_faces = faces[large_face_mask]
        large_face_ratios = face_ratios[large_face_mask]
        
        # 8% 이상 + 정면 얼굴 (신규 추가)
        six_percent_mask = face_ratios >= 8.0
        six_percent_faces = faces[six_percent_mask]
        six_percent_ratios = face_ratios[six_percent_mask]
        frontal_mask = self.is_frontal_face_advanced(six_percent_faces)
        
        # 8% 이상 얼굴 정보
        faces_8_percent_info = []
        for i, (face, ratio) in enumerate(zip(large_faces, large_face_ratios)):
            x, y, width, height = face[:4].astype(int)
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            width = max(1, min(width, w-x))
            height = max(1, min(height, h-y))
            
            face_info = {
                "face_id": int(i),
                "bbox": [int(x), int(y), int(width), int(height)],
                "confidence": float(face[14]) if len(face) > 14 else 0.0,
                "face_ratio_percent": float(round(ratio, 2)),
                "landmarks": [int(coord) for coord in face[4:14]] if len(face) >= 14 else []
            }
            faces_8_percent_info.append(face_info)
        
        # 8% 이상 + 정면 얼굴 정보
        frontal_faces_8_percent_info = []
        for i, (face, ratio) in enumerate(zip(six_percent_faces, six_percent_ratios)):
            if frontal_mask[i]:
                x, y, width, height = face[:4].astype(int)
                x = max(0, min(x, w-1))
                y = max(0, min(y, h-1))
                width = max(1, min(width, w-x))
                height = max(1, min(height, h-y))
                
                face_info = {
                    "face_id": int(i),
                    "bbox": [int(x), int(y), int(width), int(height)],
                    "confidence": float(face[14]) if len(face) > 14 else 0.0,
                    "face_ratio_percent": float(round(ratio, 2)),
                    "landmarks": [int(coord) for coord in face[4:14]] if len(face) >= 14 else []
                }
                frontal_faces_8_percent_info.append(face_info)
        
        print(f"📊 {Path(image_path).name}: 탐지된 객체 수 {len(faces)}개, 8% {len(faces_8_percent_info)}개, 8%정면 {len(frontal_faces_8_percent_info)}개")
        
        return {
            "image_path": str(image_path),
            "group_id": Path(image_path).name[:11],
            "image_count": 1,
            "image_size": [int(h), int(w)],
            "total_faces": int(len(faces)),
            "all_faces_data": all_faces_data,
            "faces_8_percent": faces_8_percent_info,
            "frontal_faces_8_percent": frontal_faces_8_percent_info
        }


    def save_visualization(self, image_path: str, result: Dict, output_dir: str):
        """우선순위 기반 단일 박스 시각화 (이미지 전용 폴더)"""
        original_image = cv2.imread(image_path)
        if original_image is None:
            return
        
        # 이미지 전용 하위 폴더 생성
        images_output_dir = os.path.join(output_dir, "visualized_images")
        os.makedirs(images_output_dir, exist_ok=True)
        
        image = original_image.copy()
        h, w = image.shape[:2]
        
        # 모든 얼굴 데이터 수집
        all_faces = result.get("all_faces_data", [])
        faces_8_percent = result.get("faces_8_percent", [])
        frontal_faces_8_percent = result.get("frontal_faces_8_percent", [])
        
        # 각 얼굴의 우선순위별 분류
        face_categories = {}  # face_id: (category, color, label_prefix)
        
        # 1. 먼저 모든 얼굴을 파란색(일반)으로 설정
        for face_info in all_faces:
            face_id = face_info["face_id"]
            face_categories[face_id] = ("general", (255, 0, 0), "DETECTED")  # 파란색
        
        # 2. 8% 이상 얼굴을 노란색으로 업데이트
        for face_info in faces_8_percent:
            face_id = face_info["face_id"]
            face_categories[face_id] = ("size", (0, 255, 255), "8% SIZE")  # 노란색
        
        # 3. 8% 정면 얼굴을 초록색으로 업데이트 (최우선)
        for face_info in frontal_faces_8_percent:
            face_id = face_info["face_id"]
            face_categories[face_id] = ("frontal", (0, 255, 0), "8% FRONTAL")  # 초록색
        
        # 4. 각 얼굴에 대해 한 번만 박스 그리기
        for face_info in all_faces:
            face_id = face_info["face_id"]
            if face_id not in face_categories:
                continue
                
            x, y, width, height = face_info["bbox"]
            confidence = face_info["confidence"]
            ratio = face_info["face_ratio_percent"]
            
            category, color, label_prefix = face_categories[face_id]
            
            # 박스 두께 설정 (우선순위별)
            thickness_map = {
                "frontal": 4,    # 8% 정면: 가장 두꺼운 선
                "size": 3,      # 8% 이상: 중간 두께
                "general": 2     # 일반: 얇은 선
            }
            thickness = thickness_map[category]
            
            # 박스 그리기
            cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness)
            
            # 라벨 배경 색상 (박스 색상의 어두운 버전)
            bg_color = tuple(int(c * 0.8) for c in color)
            
            # 라벨
            label = f"{label_prefix} #{face_id+1}"
            details = f"{ratio:.1f}% | {confidence:.2f}"
            
            # 라벨 배경
            label_height = 50
            cv2.rectangle(image, (x, y - label_height), (x + 250, y), bg_color, -1)
            
            # 라벨 텍스트
            cv2.putText(image, label, (x + 5, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, details, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 5. 범례 (우선순위 순서로 표시)
        legend_y = 120
        legend_items = [
            ("8% Frontal (Priority 1)", (0, 255, 0), 4),
            ("8% size (Priority 2)", (0, 255, 255), 3),
            ("Detected (Priority 3)", (255, 0, 0), 2)
        ]
        
        cv2.rectangle(image, (10, legend_y), (350, legend_y + 100), (50, 50, 50), -1)
        cv2.putText(image, "Legend (Priority Order):", (20, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for i, (text, color, thickness) in enumerate(legend_items):
            y_pos = legend_y + 35 + (i * 20)
            cv2.rectangle(image, (20, y_pos - 5), (40, y_pos + 5), color, thickness)
            cv2.putText(image, text, (50, y_pos + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 6. 통계 정보
        stats_lines = [
            f"File: {Path(image_path).name}",
            f"Group ID: {Path(image_path).name[:11]}",
            f"Total: {result.get('total_faces', 0)}",
            f"8% size: {len(faces_8_percent)}",
            f"8% Frontal: {len(frontal_faces_8_percent)}"
        ]
        
        # 반투명 배경
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (350, len(stats_lines) * 30 + 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
        
        for i, line in enumerate(stats_lines):
            cv2.putText(image, line, (20, 40 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 7. 우선순위별 카운트 표시
        category_counts = {"frontal": 0, "size": 0, "general": 0}
        for category, _, _ in face_categories.values():
            category_counts[category] += 1
        
        count_y = 250
        cv2.rectangle(image, (10, count_y), (280, count_y + 80), (30, 30, 30), -1)
        cv2.putText(image, "Priority Counts:", (20, count_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"Green (8% Frontal): {category_counts['frontal']}", (20, count_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(image, f"Yellow (8% size): {category_counts['size']}", (20, count_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(image, f"Blue (Detected): {category_counts['general']}", (20, count_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 파일 저장 (이미지 전용 폴더에)
        filename = Path(image_path).stem
        save_path = os.path.join(images_output_dir, f"{filename}_priority_faces.jpg")
        cv2.imwrite(save_path, image)
        print(f"📸 이미지 저장: visualized_images/{filename}_priority_faces.jpg")

    
    def save_results_csv(self, results: List[Dict], output_dir: str):
        """데이터 전용 폴더에 CSV 저장"""
        
        # 데이터 전용 하위 폴더 생성
        data_output_dir = os.path.join(output_dir, "analysis_data")
        os.makedirs(data_output_dir, exist_ok=True)
        
        grouped_data = {}
        
        for result in results:
            if "error" in result:
                continue
                
            group_id = result["group_id"]
            
            if group_id not in grouped_data:
                grouped_data[group_id] = {
                    "video_id": group_id,
                    "image_count": 0,
                    "total_faces": 0,
                    "faces_8_percent": 0,
                    "frontal_faces_8_percent": 0,
                }
            
            # 데이터 누적
            grouped_data[group_id]["image_count"] += 1
            grouped_data[group_id]["total_faces"] += result["total_faces"]
            grouped_data[group_id]["faces_8_percent"] += len(result["faces_8_percent"])
            grouped_data[group_id]["frontal_faces_8_percent"] += len(result["frontal_faces_8_percent"])
        
        df = pd.DataFrame(list(grouped_data.values()))
        
        # CSV를 데이터 전용 폴더에 저장
        csv_path = os.path.join(data_output_dir, "face_analysis_summary.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"\n📊 CSV 저장: analysis_data/face_analysis_summary.csv")
        print(f"✓ 총 그룹: {len(grouped_data)}개")
        print(f"✓ 8% 이상: {df['faces_8_percent'].sum()}개")
        print(f"✓ 8% 정면: {df['frontal_faces_8_percent'].sum()}개")
        
        return df
    
    def save_results(self, results: List[Dict], output_dir: str):
        """JSON과 CSV를 데이터 전용 폴더에 저장"""
        
        # 데이터 전용 하위 폴더 생성
        data_output_dir = os.path.join(output_dir, "analysis_data")
        os.makedirs(data_output_dir, exist_ok=True)
        
        # JSON 파일 경로를 데이터 폴더로 변경
        results_path = os.path.join(data_output_dir, "analysis_results.json")
        
        total_images = len(results)
        total_faces = sum(int(r.get("total_faces", 0)) for r in results if "error" not in r)
        total_8_percent = sum(len(r.get("faces_8_percent", [])) for r in results if "error" not in r)
        total_8_percent_frontal = sum(len(r.get("frontal_faces_8_percent", [])) for r in results if "error" not in r)
        
        summary = {
            "summary": {
                "total_images": int(total_images),
                "total_faces_detected": int(total_faces),
                "faces_8_percent": int(total_8_percent),
                "frontal_faces_8_percent": int(total_8_percent_frontal),
            },
            "detailed_results": results
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📄 JSON 저장: analysis_data/analysis_results.json")
        
        # CSV도 같은 데이터 폴더에 저장
        self.save_results_csv(results, output_dir)
    
    def process_batch_images(self, image_paths: List[str], output_dir: str = "results", save_visualizations: bool = True) -> List[Dict]:
        """배치 이미지 처리 (폴더 구조 분리)"""
        
        # 메인 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 하위 폴더들 미리 생성
        images_output_dir = os.path.join(output_dir, "visualized_images")
        data_output_dir = os.path.join(output_dir, "analysis_data")
        
        os.makedirs(images_output_dir, exist_ok=True)
        os.makedirs(data_output_dir, exist_ok=True)
        
        print(f"📁 폴더 구조 생성:")
        print(f"   └── {output_dir}/")
        print(f"       ├── visualized_images/  (이미지 저장)")
        print(f"       └── analysis_data/      (CSV, JSON 저장)")
        
        results = []
        
        progress_bar = tqdm(image_paths, desc="얼굴 분석 진행", ncols=120, unit="image")
        
        for i, image_path in enumerate(progress_bar):
            try:
                result = self.process_single_image(image_path)
                results.append(result)
                
                if save_visualizations and "error" not in result:
                    self.save_visualization(image_path, result, output_dir)
                
                if torch.cuda.is_available() and i % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                results.append({"error": str(e), "image_path": image_path})
        
        progress_bar.close()
        self.save_results(results, output_dir)
        return results

def main():
    """메인 실행 함수 (폴더 구조 분리)"""
    print("🚀 GPU 가속 얼굴 분석기 시작 (폴더 구조 분리)")
    
    analyzer = GPUFaceAnalyzer('face_detection_yunet_2023mar.onnx')
    
    # 이미지 경로 설정
    image_folder = "images"
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    
    if os.path.exists(image_folder):
        for ext in image_extensions:
            image_paths.extend(Path(image_folder).glob(f"*{ext}"))
    
    image_paths = [str(p) for p in image_paths]
    
    if image_paths:
        print(f"📸 발견된 이미지: {len(image_paths)}개")
        
        # 결과를 분리된 폴더 구조로 저장
        output_dir = "face_analysis_results"
        results = analyzer.process_batch_images(image_paths, output_dir=output_dir)
        
        print(f"\n✅ 처리 완료!")
        print(f"📁 결과 위치:")
        print(f"   ├── {output_dir}/visualized_images/     (시각화 이미지)")
        print(f"   └── {output_dir}/analysis_data/         (CSV, JSON 데이터)")
        
    else:
        print("⚠ 처리할 이미지가 없습니다.")
        print(f"💡 '{image_folder}' 폴더에 이미지를 추가하세요.")

if __name__ == "__main__":
    main()
