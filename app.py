import streamlit as st
from app_rag_system import get_qa_response

st.title("RAGチャットボット")

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# チャット履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザー入力
if prompt := st.chat_input("質問を入力してください:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ボットの応答
    with st.chat_message("assistant"):
        response = get_qa_response(prompt)
        st.markdown(response["answer"])
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

        with st.expander("参照情報"):
            st.text(response["sources"])

# サイドバーにチャット履歴のクリアボタンを追加
if st.sidebar.button("チャット履歴をクリア"):
    st.session_state.messages = []
    st.rerun()