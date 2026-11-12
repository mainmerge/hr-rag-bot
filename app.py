"""Streamlit-интерфейс HR-бота. Запуск: streamlit run app.py"""
import streamlit as st

import rag

st.set_page_config(page_title="HR-бот DemoTech", page_icon="💼")
st.title("💼 HR-бот DemoTech")
st.caption("Учебный RAG-бот: отвечает по HR-политикам вымышленной компании. "
           f"Эмбеддинги — {rag.EMBED_MODEL}, генерация — {rag.GEMINI_MODEL}.")

with st.sidebar:
    st.subheader("Примеры вопросов")
    examples = [
        "Сколько дней отпуска в год?",
        "Какие суточные в зарубежной командировке?",
        "Какой бюджет на обучение у руководителя?",
        "Сколько стоит ДМС для ребёнка?",
        "Можно ли взять собаку в офис?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.question = ex
    top_k = st.slider("Фрагментов в контексте (top-k)", 3, 10, rag.TOP_K)


@st.cache_resource(show_spinner="Загружаю модель эмбеддингов…")
def warmup():
    rag.get_embedder()
    return rag.get_collection().count()


try:
    n_vectors = warmup()
except Exception as e:  # noqa: BLE001
    st.error(f"Не удалось открыть индекс: {e}")
    st.stop()
if n_vectors == 0:
    st.warning("Индекс пуст. Выполните `python ingest.py`.")
    st.stop()

question = st.text_input("Ваш вопрос", key="question", placeholder="Например: за сколько дней подавать заявление на отпуск?")

if question:
    with st.spinner("Ищу в документах и формирую ответ…"):
        try:
            result = rag.answer(question, k=top_k)
        except Exception as e:  # noqa: BLE001
            st.error(f"Ошибка: {e}")
            st.stop()

    st.subheader("Ответ")
    if result.found:
        st.success(result.text)
    else:
        st.info(result.text)

    st.subheader("Источники")
    if result.sources:
        titles = {h.source: h.title for h in result.hits}
        for s in result.sources:
            st.markdown(f"- **{titles.get(s, s)}** — `docs/{s}`")
    else:
        st.write("—")

    with st.expander("Найденные фрагменты (retrieval)"):
        for i, h in enumerate(result.hits, 1):
            st.markdown(f"**{i}. {h.source}** · сходство {h.score:.3f}")
            st.text(h.text)
