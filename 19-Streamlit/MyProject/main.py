import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
import glob
from langchain_teddynote import logging

# from langchain import hub
# from langchain_teddynote.messages import stream_response


# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project]PDF RAG")

st.title("나만의 채찍PT 테스트😋")


# 처음 한번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도
    st.session_state["messages"] = []

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    prompt_files = glob.glob("prompts/*.yaml")
    selected_prompt = st.selectbox("프롬프트를 선택해주세요", prompt_files, index=0)
    task_input = st.text_input("TASK 입력", "")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 체인 생성
def create_chain(prompt_filepath, task=""):
    # prompt 적용
    prompt = load_prompt(prompt_filepath, encoding="utf-8")
    # 추가 파라미터가 있으면 추가
    if task:
        prompt = prompt.partial(task=task)

    # GPT
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    # 출력 파서
    output_parser = StrOutputParser()

    # 체인 생성
    chain = prompt | llm | output_parser
    return chain


# 초기화 버튼이 눌리면
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력력
print_messages()


# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어봐~")

# 만약 사용자 입력이 들어오면
if user_input:
    # 사용자의 입력
    st.chat_message("user").write(user_input)
    # chain을 생성
    chain = create_chain(selected_prompt, task=task_input)

    response = chain.stream({"question": user_input})
    with st.chat_message("assistant"):
        # 빈 공간(컨테이너)을 만들어서 여기에 토큰을 스트리밍 출력한다.
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # 대화기록을 저장한다.
    add_message("user", user_input)
    add_message("assistant", ai_answer)
