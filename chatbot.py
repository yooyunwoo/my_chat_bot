from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI
from openai import OpenAIError

# 설정은 시스템 환경 변수가 아닌 이 파일 옆의 .env 에서만 읽는다.
ENV_PATH = Path(__file__).resolve().parent / ".env"
CONFIG = dotenv_values(ENV_PATH)


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
            "content": "You are a helpful Korean-speaking assistant.",
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
