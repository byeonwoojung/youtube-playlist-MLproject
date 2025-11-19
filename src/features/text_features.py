"""
제목(Title) 피처 추출 모듈

YouTube 영상 제목에서 다음 피처를 추출합니다:
1. 기본 피처 (emoji 라이브러리 사용):
   - has_question_or_exclamation: 물음표/느낌표 포함 여부 (0 또는 1)
   - char_length: 공백 제외 문자 수
   - has_emoji: 이모지/기호/카오모지 포함 여부 (0 또는 1)
   - emoji_ratio: 이모지/기호/카오모지 비율 (0.0 ~ 1.0)

2. OpenAI API 피처:
   - attention_score: 주목도 점수 (0.0 ~ 1.0)
   - sensory: 오감 자극 표현 여부 (0: 오감 없음, 1: 오감 있음)
   - genre_mentioned: 음악 장르 언급 여부 (0: 언급 없음, 1: 언급 있음)
"""

import pandas as pd
import emoji
import re
import html
from unidecode import unidecode
from typing import Optional, Tuple
import os
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.helpers import load_csv_safely, save_csv_safely
except ImportError:
    def load_csv_safely(filepath, encoding="utf-8"):
        return pd.read_csv(filepath, encoding=encoding)
    
    def save_csv_safely(df, filepath, encoding="utf-8-sig"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False, encoding=encoding)


class TitleFeatureExtractor:
    """제목 피처 추출 클래스"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Args:
            openai_api_key: OpenAI API 키 (선택사항)
        """
        self.openai_api_key = openai_api_key
        self.openai_client = None
        
        if openai_api_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
            except ImportError:
                print("⚠️  OpenAI 라이브러리가 설치되지 않았습니다. pip install openai")
    
    # ========== 기본 피처 추출 함수 ==========
    
    @staticmethod
    def has_question_or_exclamation(title: str) -> int:
        """
        물음표 또는 느낌표 여부 (있으면 1, 없으면 0)
        
        Args:
            title: 영상 제목
            
        Returns:
            0 또는 1
        """
        if pd.isna(title):
            return 0
        title = str(title).strip()
        return int(('?' in title) or ('!' in title))
    
    @staticmethod
    def has_emoji_or_symbol(title: str) -> bool:
        """
        이모지 + 기호 + 카오모지 포함 여부 (있으면 True, 없으면 False)
        
        Args:
            title: 영상 제목
            
        Returns:
            True 또는 False
        """
        if pd.isna(title):
            return False
        
        title = str(title)
        
        # 이모지 확인
        if any(char in emoji.EMOJI_DATA for char in title):
            return True
        
        # 기호 확인 (하트, 음표 등)
        if re.search(r'[\u2600-\u26FF\u2700-\u27BF♡♪]', title):
            return True
        
        # 카오모지 확인
        if re.search(r'[\(\[][^\)\]]{1,15}[\)\]]', title):
            return True
        
        return False
    
    @staticmethod
    def char_length_no_space(title: str) -> int:
        """
        공백 제외 문자 수
        
        Args:
            title: 영상 제목
            
        Returns:
            공백 제외 문자 수
        """
        if pd.isna(title):
            return 0
        return len(str(title).replace(" ", ""))
    
    @staticmethod
    def emoji_symbol_ratio(title: str) -> float:
        """
        이모지/기호/카오모지 비율 (공백 제외 문자 수 기준)
        
        Args:
            title: 영상 제목
            
        Returns:
            비율 (0.0 ~ 1.0)
        """
        if pd.isna(title):
            return 0.0
        
        title = str(title)
        text_no_space = title.replace(" ", "")
        total_len = len(text_no_space)
        
        if total_len == 0:
            return 0.0
        
        # 이모지 개수
        emoji_count = sum(char in emoji.EMOJI_DATA for char in text_no_space)
        
        # 기호 개수
        symbol_count = len(re.findall(r'[\u2600-\u26FF\u2700-\u27BF♡♪]', text_no_space))
        
        # 카오모지 개수
        kaomoji_count = len(re.findall(r'[\(\[][^\)\]]{1,15}[\)\]]', text_no_space))
        
        total_special = emoji_count + symbol_count + kaomoji_count
        
        return total_special / total_len
    
    # ========== 한글 전처리 함수 ==========
    
    @staticmethod
    def safe_unidecode(text: str) -> str:
        """
        안전한 유니코드 디코딩 (한글, 숫자, 공백 유지)
        
        Args:
            text: 원본 텍스트
            
        Returns:
            디코딩된 텍스트
        """
        result = ""
        for c in text:
            if '\uAC00' <= c <= '\uD7A3':  # 한글
                result += c
            elif c.isdigit() or c.isspace():  # 숫자, 공백 유지
                result += c
            elif c.isascii() or not c.isprintable():
                result += c
            else:
                result += unidecode(c)
        return result
    
    @staticmethod
    def clean_title_korean_only_strict(title: str) -> str:
        """
        한글만 남겨둔 채로 제목 정제 (OpenAI API 사용을 위한 전처리)
        
        Args:
            title: 원본 제목
            
        Returns:
            정제된 제목 (한글, 숫자, 공백, ?, !, . 만 포함)
        """
        if pd.isna(title):
            return ""
        
        title = str(title)
        title = html.unescape(title)
        title = TitleFeatureExtractor.safe_unidecode(title)
        
        # playlist tag 제거
        playlist_patterns = [
            r'\[.*?playlist.*?\]', r'\(.*?playlist.*?\)',
            r'playlist', r'Playlist', r'PLAYLIST'
        ]
        for pattern in playlist_patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        # 영어 제거
        title = re.sub(r'[A-Za-z]', '', title)
        
        # 이모지 및 모든 특수문자 제거 (한글, 숫자, 공백, ?, !, . 제외)
        title = re.sub(r'[^\uAC00-\uD7A30-9\s\?\.\!]', '', title)
        
        # 공백 정리
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title
    
    # ========== OpenAI API 피처 추출 ==========
    
    def make_combined_prompt(self, text: str) -> str:
        """
        주목도(attention_score) + 오감(sensory) + 장르(genre_mentioned) 통합 프롬프트 생성
        
        Args:
            text: 정제된 제목
            
        Returns:
            OpenAI API 프롬프트
        """
        return f"""
당신은 유튜브 제목 분석 전문가입니다. 아래 문장을 읽고 다음 세 가지 항목을 판단하세요:

[1] 주목도 점수 평가 (attention score):
이 제목이 사람들의 시선을 얼마나 끄는지, 클릭을 얼마나 유도하는지를 0.0~1.0 사이 숫자로 수치화하세요.

판단 기준:
- 점수가 높을수록 자극적이며 클릭을 유도합니다.
- 점수가 낮을수록 정보성, 나열형 제목이며 흥미 요소가 적습니다.

[높은 점수 예시 (0.8 ~ 1.0)]
- "1초 만에 반하게 되는 감성 BGM"
- "지금 안 보면 놓칩니다! 역대급 조합"
- "미쳤다 진짜... 이런 건 처음 들어봐"
- "출근길에 이거 하나면 됨!"

[중간 점수 예시 (0.4 ~ 0.7)]
- "잔잔한 저녁, 감성 짙은 노래 모음"
- "비 오는 날 듣기 좋은 감성 재즈"
- "봄에 어울리는 따뜻한 음악 리스트"
- "퇴근길을 위로해주는 음악들"

[낮은 점수 예시 (0.0 ~ 0.3)]
- "클래식, 재즈, 팝, 락 장르별 모음"
- "감성 힐링 BGM 리스트"
- "겨울 분위기의 재즈 플레이리스트"
- "편안한 분위기의 음악 추천"

[2] 오감 자극 표현 판별:
문장에 시각, 촉각, 후각, 미각 중 하나라도 자극하는 단어가 포함되어 있는지 판단하세요.

판단 기준:
(1) 시각: 눈에 보이는 감각 표현 – 예: 다채로운, 화려한, 빛나는, 어두운
(2) 촉각: 만지는 감각 – 예: 따뜻한, 포근한, 촉촉한
(3) 후각: 냄새 – 예: 향긋한, 고소한 냄새, 스모키한 향
(4) 미각: 맛 – 예: 달콤한, 고소한 맛, 새콤한, 아삭한

※ 청각(소리, 음악, 듣기 등)은 "오감 없음"

[3] 음악 장르 언급 여부 (genre_mentioned):
다음 음악 장르 중 하나라도 언급되면 "언급 있음", 그렇지 않으면 "언급 없음"으로 판단하세요.
- 재즈, 클래식, 팝, 락, 힙합, 발라드, 알앤비(R&B), 블루스, 포크, 인디, 트로트, EDM, 일렉트로닉, 하우스, 뉴에이지

※ 반드시 아래 형식으로 응답하세요:
attention_score: (0.0~1.0의 숫자만)
오감: 오감 있음 / 오감 없음
장르: 언급 있음 / 언급 없음

문장:
{text}

당신의 판단:
"""
    
    def analyze_title_with_openai(self, text: str) -> Tuple[Optional[float], Optional[int], Optional[int]]:
        """
        OpenAI API를 사용하여 제목 분석
        
        Args:
            text: 정제된 제목
            
        Returns:
            (attention_score, sensory, genre_mentioned)
            - attention_score: 0.0 ~ 1.0 또는 None
            - sensory: 0 (오감 없음) 또는 1 (오감 있음) 또는 None
            - genre_mentioned: 0 (언급 없음) 또는 1 (언급 있음) 또는 None
        """
        if not self.openai_client:
            print("⚠️  OpenAI API 클라이언트가 초기화되지 않았습니다.")
            return None, None, None
        
        try:
            prompt = self.make_combined_prompt(text)
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            output = response.choices[0].message.content.strip()
            lines = output.split("\n")
            
            attention_score, sensory, genre = None, None, None
            
            for line in lines:
                if line.lower().startswith("attention_score:"):
                    try:
                        attention_score = float(line.split(":")[1].strip())
                    except:
                        attention_score = None
                elif line.lower().startswith("오감:"):
                    sensory_text = line.split(":")[1].strip()
                    sensory = 1 if "오감 있음" in sensory_text else 0
                elif line.lower().startswith("장르:"):
                    genre_text = line.split(":")[1].strip()
                    genre = 1 if "언급 있음" in genre_text else 0
            
            return attention_score, sensory, genre
        
        except Exception as e:
            print(f"❗Error analyzing title: {e}")
            return None, None, None
    
    # ========== 전체 피처 추출 ==========
    
    def extract_basic_features(self, df: pd.DataFrame, title_column: str = 'title') -> pd.DataFrame:
        """
        기본 피처 추출 (emoji 라이브러리 사용)
        
        Args:
            df: 원본 DataFrame
            title_column: 제목 컬럼명
            
        Returns:
            기본 피처가 추가된 DataFrame
        """
        print("\n" + "=" * 60)
        print("기본 제목 피처 추출 시작".center(60))
        print("=" * 60)
        
        result_df = df.copy()
        
        # 1. 물음표/느낌표 여부
        print("\n[1/4] 물음표/느낌표 여부(has_question_or_exclamation) 추출 중...")
        result_df['has_question_or_exclamation'] = result_df[title_column].apply(
            self.has_question_or_exclamation
        )
        count = result_df['has_question_or_exclamation'].sum()
        print(f"  ✓ 물음표/느낌표 포함: {count}/{len(result_df)} ({count/len(result_df)*100:.1f}%)")
        
        # 2. 공백 제외 문자 수
        print("\n[2/4] 글자 수(char_length) 추출 중...")
        result_df['char_length'] = result_df[title_column].apply(self.char_length_no_space)
        print(f"  ✓ 평균 글자 수: {result_df['char_length'].mean():.2f}")
        
        # 3. 이모지 포함 여부
        print("\n[3/4] 이모지 포함 여부(has_emoji) 확인 중...")
        result_df['has_emoji'] = result_df[title_column].apply(self.has_emoji_or_symbol)
        result_df['has_emoji'] = result_df['has_emoji'].astype(int)
        emoji_count = result_df['has_emoji'].sum()
        print(f"  ✓ 이모지 포함: {emoji_count}/{len(result_df)} ({emoji_count/len(result_df)*100:.1f}%)")
        
        # 4. 이모지 비율
        print("\n[4/4] 이모지 비율(emoji_ratio) 계산 중...")
        result_df['emoji_ratio'] = result_df[title_column].apply(self.emoji_symbol_ratio)
        print(f"  ✓ 평균 이모지 비율: {result_df['emoji_ratio'].mean():.4f}")
        
        print("\n" + "=" * 60)
        print("✅ 기본 피처 추출 완료".center(60))
        print("=" * 60)
        
        return result_df
    
    def extract_openai_features(
        self, 
        df: pd.DataFrame, 
        title_column: str = 'title',
        save_interval: int = 500,
        output_prefix: str = "progressive_save"
    ) -> pd.DataFrame:
        """
        OpenAI API를 사용한 피처 추출
        
        Args:
            df: 원본 DataFrame (cleaned_title 컬럼 포함)
            title_column: 정제된 제목 컬럼명 (기본값: 'cleaned_title')
            save_interval: 중간 저장 주기
            output_prefix: 중간 저장 파일명 접두사
            
        Returns:
            OpenAI 피처가 추가된 DataFrame
        """
        if not self.openai_client:
            print("⚠️  OpenAI API 키가 설정되지 않았습니다. 이 단계를 건너뜁니다.")
            return df
        
        print("\n" + "=" * 60)
        print("OpenAI API 피처 추출 시작".center(60))
        print("=" * 60)
        print(f"모델: gpt-4o-mini")
        print(f"중간 저장 주기: {save_interval}행마다")
        
        result_df = df.copy()
        
        # cleaned_title이 없으면 생성
        if 'cleaned_title' not in result_df.columns:
            print("\n[전처리] cleaned_title 생성 중...")
            result_df['cleaned_title'] = result_df[title_column].apply(
                self.clean_title_korean_only_strict
            )
        
        # 피처 초기화
        result_df['attention_score'] = None
        result_df['sensory'] = None
        result_df['genre_mentioned'] = None
        
        # 배치 처리
        for i in tqdm(range(len(result_df)), desc="OpenAI API 호출"):
            text = result_df.loc[i, 'cleaned_title']
            score, sensory, genre = self.analyze_title_with_openai(text)
            
            result_df.loc[i, 'attention_score'] = score
            result_df.loc[i, 'sensory'] = sensory
            result_df.loc[i, 'genre_mentioned'] = genre
            
            # 중간 저장
            if (i + 1) % save_interval == 0 or (i + 1) == len(result_df):
                file_index = (i + 1) // save_interval
                filename = f"{output_prefix}_{file_index:04d}.csv"
                result_df.iloc[:i+1].to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"\n✅ 중간 저장: {i+1}행까지 → {filename}")
        
        print("\n" + "=" * 60)
        print("✅ OpenAI 피처 추출 완료".center(60))
        print("=" * 60)
        
        return result_df
    
    def extract_all_features(
        self, 
        df: pd.DataFrame, 
        title_column: str = 'title',
        use_openai: bool = False,
        save_interval: int = 500
    ) -> pd.DataFrame:
        """
        모든 제목 피처 일괄 추출
        
        Args:
            df: 원본 DataFrame
            title_column: 제목 컬럼명
            use_openai: OpenAI API 사용 여부
            save_interval: OpenAI 중간 저장 주기
            
        Returns:
            모든 피처가 추가된 DataFrame
        """
        # 1. 기본 피처 추출
        result_df = self.extract_basic_features(df, title_column)
        
        # 2. OpenAI 피처 추출 (선택사항)
        if use_openai and self.openai_client:
            result_df = self.extract_openai_features(
                result_df, 
                title_column='cleaned_title',
                save_interval=save_interval
            )
        
        return result_df


def extract_title_features(
    input_csv: str,
    output_csv: str,
    title_column: str = 'title',
    openai_api_key: Optional[str] = None,
    use_openai: bool = False,
    save_interval: int = 500
) -> pd.DataFrame:
    """
    제목 피처 추출 메인 함수
    
    Args:
        input_csv: 입력 CSV 경로
        output_csv: 출력 CSV 경로
        title_column: 제목 컬럼명
        openai_api_key: OpenAI API 키 (선택사항)
        use_openai: OpenAI API 사용 여부
        save_interval: OpenAI 중간 저장 주기
        
    Returns:
        피처가 추가된 DataFrame
    """
    # 데이터 로드
    print(f"\n📂 데이터 로드: {input_csv}")
    df = load_csv_safely(input_csv)
    print(f"  ✓ {len(df)} rows 로드 완료")
    
    # 피처 추출
    extractor = TitleFeatureExtractor(openai_api_key=openai_api_key)
    df_result = extractor.extract_all_features(
        df, 
        title_column=title_column,
        use_openai=use_openai,
        save_interval=save_interval
    )
    
    # 최종 정리: 필요한 컬럼만 선택
    columns_to_keep = [
        'video_id',
        'char_length',
        'has_emoji',
        'emoji_ratio'
    ]
    
    if use_openai and 'attention_score' in df_result.columns:
        columns_to_keep.extend(['attention_score', 'sensory', 'genre_mentioned'])
    
    df_final = df_result[columns_to_keep]
    
    # 저장
    save_csv_safely(df_final, output_csv)
    print(f"\n💾 결과 저장: {output_csv}")
    
    return df_final


if __name__ == "__main__":
    # 테스트 실행
    INPUT_CSV = "../rawData/youtubeInfo/allYoutubeInfo_themeFiltered.csv"
    OUTPUT_CSV = "../rawData/titles/titles_final.csv"
    
    # OpenAI API 키 (환경변수 또는 직접 입력)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
    
    df = extract_title_features(
        INPUT_CSV, 
        OUTPUT_CSV,
        openai_api_key=OPENAI_API_KEY,
        use_openai=bool(OPENAI_API_KEY)  # API 키가 있으면 사용
    )
    
    print("\n📊 추출된 피처:")
    print(df.head(10))
