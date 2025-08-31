from App.utils import get_llm_model,get_retrever,get_prompt
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser


retriver = get_retrever()
llm = get_llm_model()
prompt = get_prompt()


chain = (
    {
        "documents": RunnableLambda(lambda x: "\n\n".join(doc.page_content for doc in retriver.invoke(x))),
        "query": RunnableLambda(lambda x: x),
    }
    |
    prompt |
    llm |
    StrOutputParser ()

)

print(chain.invoke("what is the director of the clinic"))







