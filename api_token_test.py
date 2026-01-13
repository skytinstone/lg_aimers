"""
API Token 통합 테스트 스크립트
- Hugging Face Token
- Friendli AI API Key
"""

import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
import requests

# 환경 변수 로드
load_dotenv()


def test_hf_token():
    """Hugging Face Token 테스트"""
    
    print("\n" + "="*60)
    print("1. Hugging Face Token 테스트")
    print("="*60)
    
    # 1. .env 파일에서 토큰 읽기
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("❌ HF_TOKEN이 .env 파일에 설정되지 않았습니다!")
        return False
    
    print(f"✓ HF_TOKEN 발견: {hf_token[:10]}...{hf_token[-5:]}")
    
    # 2. 토큰 형식 확인
    if not hf_token.startswith("hf_"):
        print("⚠️  경고: 토큰이 'hf_'로 시작하지 않습니다. 올바른 토큰인지 확인하세요.")
        return False
    
    print("✓ 토큰 형식 확인 완료")
    
    # 3. Hugging Face API로 토큰 유효성 검증
    try:
        print("\n토큰 유효성 검증 중...")
        api = HfApi()
        user_info = api.whoami(token=hf_token)
        
        print(f"✓ 토큰 인증 성공!")
        print(f"  - 사용자: {user_info['name']}")
        print(f"  - 이메일: {user_info.get('email', 'N/A')}")
        print(f"  - 타입: {user_info.get('type', 'N/A')}")
        
    except Exception as e:
        print(f"❌ 토큰 인증 실패: {str(e)}")
        print("\n다음 사항을 확인하세요:")
        print("1. https://huggingface.co/settings/tokens 에서 토큰을 확인하세요")
        print("2. 토큰이 만료되지 않았는지 확인하세요")
        print("3. 토큰에 적절한 권한이 있는지 확인하세요")
        return False
    
    # 4. EXAONE 모델 접근 권한 테스트
    try:
        print("\nEXAONE 모델 접근 권한 확인 중...")
        model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"
        
        model_info = api.model_info(model_name, token=hf_token)
        print(f"✓ EXAONE 모델 접근 가능!")
        print(f"  - 모델: {model_info.id}")
        print(f"  - 다운로드 수: {model_info.downloads:,}")
        print(f"  - 라이선스: {model_info.cardData.get('license', 'N/A') if model_info.cardData else 'N/A'}")
        
    except Exception as e:
        print(f"⚠️  EXAONE 모델 접근 실패: {str(e)}")
        print("\n다음 사항을 확인하세요:")
        print("1. https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B 에 접속")
        print("2. 모델 라이선스 동의가 필요할 수 있습니다")
        print("3. 로그인 후 'Agree and access repository' 클릭")
        return False
    
    # 5. 로그인 시도
    try:
        print("\nHugging Face Hub 로그인 시도...")
        login(token=hf_token, add_to_git_credential=False)
        print("✓ 로그인 성공!")
        
    except Exception as e:
        print(f"⚠️  로그인 경고: {str(e)}")
    
    print("\n✅ Hugging Face Token 테스트 통과!")
    return True


def test_friendli_api():
    """Friendli AI API Key 테스트"""
    
    print("\n" + "="*60)
    print("2. Friendli AI API Key 테스트")
    print("="*60)
    
    # 1. .env 파일에서 API Key 읽기
    friendli_key = os.getenv("FRIENDLI_API_KEY")
    
    if not friendli_key:
        print("❌ FRIENDLI_API_KEY가 .env 파일에 설정되지 않았습니다!")
        return False
    
    print(f"✓ FRIENDLI_API_KEY 발견: {friendli_key[:10]}...{friendli_key[-5:]}")
    
    # 2. API Key 형식 확인
    if not friendli_key.startswith("flp_"):
        print("⚠️  경고: API Key가 'flp_'로 시작하지 않습니다. 올바른 키인지 확인하세요.")
    
    print("✓ API Key 형식 확인 완료")
    
    # 3. Friendli AI API 엔드포인트 테스트 (여러 엔드포인트 시도)
    api_success = False
    
    try:
        print("\nFriendli AI API 연결 테스트 중...")
        
        headers = {
            "Authorization": f"Bearer {friendli_key}",
            "Content-Type": "application/json"
        }
        
        # 시도 1: /dedicated-endpoints 엔드포인트
        print("  시도 1: Dedicated Endpoints 조회...")
        base_url = "https://api.friendli.ai/dedicated"
        response = requests.get(
            f"{base_url}/v1/dedicated-endpoints",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            print("  ✓ API 인증 성공! (Dedicated Endpoints)")
            data = response.json()
            if 'data' in data:
                print(f"    - Endpoints: {len(data['data'])}개")
            api_success = True
            
        elif response.status_code == 401:
            print("  ❌ 인증 실패: 유효하지 않은 API Key입니다.")
            return False
            
        elif response.status_code == 403:
            print("  ❌ 접근 권한 없음: API Key에 필요한 권한이 없습니다.")
            return False
            
        elif response.status_code == 404:
            print("  ⚠️  엔드포인트를 찾을 수 없음 (다른 방법 시도)")
            
            # 시도 2: /serverless-endpoints 엔드포인트
            print("  시도 2: Serverless Endpoints 조회...")
            response = requests.get(
                f"{base_url}/v1/serverless-endpoints",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                print("  ✓ API 인증 성공! (Serverless Endpoints)")
                data = response.json()
                if 'data' in data:
                    print(f"    - Endpoints: {len(data['data'])}개")
                api_success = True
                
            elif response.status_code == 404:
                print("  ⚠️  엔드포인트를 찾을 수 없음 (다른 방법 시도)")
                
                # 시도 3: OpenAI 호환 API로 간단한 요청
                print("  시도 3: Chat Completions API 테스트...")
                response = requests.post(
                    "https://inference.friendli.ai/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": "meta-llama-3.1-8b-instruct",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 5
                    },
                    timeout=15
                )
                
                if response.status_code == 200:
                    print("  ✓ API 인증 성공! (Chat Completions)")
                    api_success = True
                elif response.status_code == 401:
                    print("  ❌ 인증 실패: 유효하지 않은 API Key입니다.")
                    return False
                elif response.status_code == 403:
                    print("  ❌ 접근 권한 없음")
                    return False
                else:
                    print(f"  ⚠️  예상치 못한 응답: HTTP {response.status_code}")
                    print(f"     응답: {response.text[:200]}")
            
            elif response.status_code == 401:
                print("  ❌ 인증 실패: 유효하지 않은 API Key입니다.")
                return False
        else:
            print(f"  ⚠️  예상치 못한 응답: HTTP {response.status_code}")
            print(f"     응답: {response.text[:200]}")
            
    except requests.exceptions.Timeout:
        print("❌ API 요청 시간 초과: 네트워크 연결을 확인하세요.")
        return False
        
    except requests.exceptions.ConnectionError:
        print("❌ 연결 실패: 인터넷 연결을 확인하세요.")
        return False
        
    except Exception as e:
        print(f"❌ API 테스트 실패: {str(e)}")
        return False
    
    if not api_success:
        print("\n❌ Friendli AI API 테스트 실패!")
        print("\n다음 사항을 확인하세요:")
        print("1. https://suite.friendli.ai/ 에서 로그인")
        print("2. Settings > API Keys 메뉴에서 키 확인")
        print("3. 키가 활성화되어 있는지 확인")
        print("4. API 사용량 할당량이 남아있는지 확인")
        return False
    
    # 4. 사용 가능한 모델 조회 (선택적)
    try:
        print("\n사용 가능한 모델 확인 중...")
        response = requests.get(
            "https://inference.friendli.ai/v1/models",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            models_data = response.json()
            print(f"✓ 사용 가능한 모델 목록 조회 성공!")
            
            if 'data' in models_data and models_data['data']:
                print(f"  - 총 {len(models_data['data'])}개 모델 사용 가능")
                
                # EXAONE 모델 찾기
                exaone_models = [m for m in models_data['data'] 
                                if 'exaone' in m.get('id', '').lower()]
                
                if exaone_models:
                    print(f"  - EXAONE 모델 {len(exaone_models)}개 발견:")
                    for model in exaone_models[:5]:
                        print(f"    * {model.get('id', 'N/A')}")
                else:
                    print("  ⚠️  EXAONE 모델을 찾을 수 없습니다.")
                    print("     (별도 배포 설정이 필요할 수 있습니다)")
            else:
                print("  ⚠️  모델 목록이 비어있습니다.")
        else:
            print(f"  ⚠️  모델 목록 조회 실패 (HTTP {response.status_code})")
            # 모델 목록 조회 실패는 치명적이지 않으므로 계속 진행
            
    except Exception as e:
        print(f"  ⚠️  모델 확인 중 오류: {str(e)}")
    
    print("\n✅ Friendli AI API Key 테스트 통과!")
    return True


def main():
    """메인 테스트 함수"""
    
    print("="*60)
    print("API Token 통합 테스트")
    print("="*60)
    
    results = {
        "huggingface": False,
        "friendli": False
    }
    
    # Hugging Face 테스트
    try:
        results["huggingface"] = test_hf_token()
    except Exception as e:
        print(f"\n❌ Hugging Face 테스트 중 오류 발생: {str(e)}")
    
    # Friendli AI 테스트
    try:
        results["friendli"] = test_friendli_api()
    except Exception as e:
        print(f"\n❌ Friendli AI 테스트 중 오류 발생: {str(e)}")
    
    # 최종 결과
    print("\n" + "="*60)
    print("테스트 결과 요약")
    print("="*60)
    print(f"Hugging Face Token: {'✅ 통과' if results['huggingface'] else '❌ 실패'}")
    print(f"Friendli AI API Key: {'✅ 통과' if results['friendli'] else '❌ 실패'}")
    print("="*60)
    
    if all(results.values()):
        print("\n🎉 모든 API 테스트 통과! 양자화 작업을 시작할 수 있습니다!")
        print("\n다음 단계:")
        print("  python exaone_quantization.py")
        return True
    else:
        print("\n⚠️  일부 API 테스트 실패. .env 파일과 API 키를 확인하세요.")
        print("\n.env 파일 형식:")
        print("  HF_TOKEN=hf_xxxxxxxxxxxxx")
        print("  FRIENDLI_API_KEY=flp_xxxxxxxxxxxxx")
        
        if results['huggingface'] and not results['friendli']:
            print("\n참고: Hugging Face만으로도 양자화 작업은 가능합니다.")
            print("     Friendli AI는 선택적 기능입니다.")
        
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)