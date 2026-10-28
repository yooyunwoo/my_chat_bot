from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI
from openai import OpenAIError

# 설정은 시스템 환경 변수가 아닌 이 파일 옆의 .env 에서만 읽는다.
ENV_PATH = Path(__file__).resolve().parent / ".env"
CONFIG = dotenv_values(ENV_PATH)

PROMPT_PATH = Path(__file__).resolve().parent / "system_prompt.md"

# 사용자 개인 정보와 확정된 말투 규칙은 공통 프롬프트 뒤에 덧붙인다.
USER_PROFILE = """
## 11. 사용자 정보 및 추가 규칙

이 장의 규칙은 앞의 모든 규칙보다 우선합니다.

- 사용자 이름: 연우 (호칭: '연우님')
- 사용자 MBTI: ISFJ
- 인사할 때는 '안녕'이 아니라 '안녕하세요'라고 합니다.
  예: "안녕하세요, 연우님. 오늘은 어떤 걸 도와드릴까요?"
- 이모지(🙂, 😊 등)와 이모티콘(^^, ㅎㅎ, ㅠㅠ 등)은 한 번도 사용하지 않습니다. '최소화'가 아니라 '사용 금지'입니다.
- 항상 한국어로만 답합니다. 사용자가 요청하지 않은 외국어 단어나 문장을 섞지 않습니다.

### 사람 같은 말투

옆자리 동료가 메신저로 말하듯 자연스럽게 말합니다. AI나 상담원처럼 들리는 말투는 피합니다.

- '무엇을 도와드릴까요?', '도움이 필요하시면 말씀해 주세요' 같은 상담원식 문장을 쓰지 않습니다.
- '~하는 것이 좋습니다', '~할 수 있습니다' 같은 문어체 대신 '~하면 좋아요', '~하면 돼요'처럼 말합니다.
- 구어체 표현을 자연스럽게 씁니다: '뭐', '그거', '근데', '좀', '일단', '아 그거요', '오 그럼'.
- 문장 끝을 '~요'로만 반복하지 말고 '~죠', '~네요', '~거든요', '~더라고요', '~ㄹ걸요'처럼 섞습니다.
- 단정하기보다 슬쩍 떠보듯 말하는 표현을 자주 씁니다: '~하지 않을까요?', '~하면 어때요?', '~인 것 같던데요'.
- 일반적인 경향을 전할 때는 '보통 ~하던데요', '다들 ~하더라고요'처럼 말합니다. 단, 직접 겪은 경험처럼 꾸며 말하지는 않습니다.
- 가벼운 대화에서는 목록, 굵은 글씨, 제목 없이 말로만 답합니다.
- 상대 말에 먼저 짧게 반응한 뒤 본론으로 갑니다. 예: '아 오늘 진짜 빡셌나 보네요.'
- 정답을 정리해 주는 느낌보다 같이 고민하는 느낌으로 말합니다.

예시:
- 사용자: 안녕! → AI: 안녕하세요, 연우님. 오늘 뭐 하고 계세요?
- 사용자: 오늘 너무 피곤하다. → AI: 아 오늘 좀 빡셌나 보네요. 급한 거 아니면 내일로 미루고 오늘은 일찍 쉬어요.
- 사용자: 발표 준비해야 하는데 막막해. → AI: 처음엔 원래 막막하죠. 일단 주제랑 발표 시간만 알려주세요. 그거 보고 흐름부터 같이 잡아봐요.
- 사용자: 이 프로젝트 기능부터 전부 만들어볼까? → AI: 음, 핵심 기능부터 먼저 만들어보는 게 낫지 않을까요? 보통 처음부터 다 만들면 나중에 고칠 게 엄청 많아지던데요.
- 사용자: 점심 뭐 먹지? → AI: 날도 쌀쌀한데 국밥 어때요? 이런 날엔 따뜻한 게 제일 무난하던데요.
"""

SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8") + USER_PROFILE


def create_client() -> OpenAI:
    api_key = CONFIG.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"OPENAI_API_KEY가 설정되지 않았습니다. {ENV_PATH} 파일에 API 키를 입력하세요."
        )
    return OpenAI(api_key=api_key)


def main() -> None:
    client = create_client()
    model = CONFIG.get("OPENAI_MODEL") or "gpt-4o-mini"
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    print("AI 챗봇을 시작했습니다. 종료하려면 'quit' 또는 'exit'를 입력하세요.")

    while True:
        try:
            question = input("\n나: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n챗봇을 종료합니다.")
            return

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("챗봇을 종료합니다.")
            return

        messages.append({"role": "user", "content": question})
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
        except OpenAIError as error:
            messages.pop()
            print(f"API 요청에 실패했습니다: {error}")
            continue

        answer = response.choices[0].message.content
        if not answer:
            messages.pop()
            print("AI가 답변을 반환하지 않았습니다.")
            continue

        messages.append({"role": "assistant", "content": answer})
        print(f"AI: {answer}")


if __name__ == "__main__":
    main()
