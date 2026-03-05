from typing import Any
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableParallel

from src.llm_factory import get_llm


SYSTEM_PROMPT = """Eres un asistente de soporte técnico especializado. Tu objetivo es ayudar a los usuarios respondiendo preguntas sobre la documentación técnica proporcionada.

INSTRUCCIONES IMPORTANTES:
1. Solo debes responder basándote en el contexto proporcionado en la sección "Contexto".
2. Si la información necesaria para responder NO está en el contexto, debes admitirlo claramente diciendo: "No tengo información suficiente en la documentación para responder esta pregunta."
3. NO inventes ni猜测es información que no esté explícitamente en el contexto.
4. Sé conciso pero completo en tus respuestas.
5. Si el contexto contiene múltiples fuentes relevantes, combínalas de manera coherente.
6. Cite las fuentes cuando sea apropiado indicando de qué documento proviene la información.

RESPUESTA:
"""

HUMAN_PROMPT = """Contexto:
{context}

Pregunta del usuario: {question}

Responde basándote únicamente en el contexto proporcionado."""


def build_qa_chain(retriever: Any):
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": prompt,
        },
        return_source_documents=True,
        verbose=False,
    )

    return qa_chain


def build_runnable_qa_chain(retriever: Any):
    llm = get_llm()

    prompt = PromptTemplate.from_template(
        SYSTEM_PROMPT + "\n\nContexto:\n{context}\n\nPregunta: {question}\n\nRespuesta:"
    )

    runnable = RunnableParallel(
        context=retriever,
        question=lambda x: x["question"],
    ) | prompt | llm

    return runnable
