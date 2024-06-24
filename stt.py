import sys
import whisper
import pysqlite3
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.passthrough import RunnablePick



llm = Ollama(model="llama3")

def transcribe(file):
	model = whisper.load_model("large-v3.pt")
	result = model.transcribe(file)
	f = open(file.split('.')[0]+'.txt','w+')
	f.write(result["text"])
	f.close()

def rag(file):
	text_splitter = RecursiveCharacterTextSplitter(
		separators=[".",";"],
		chunk_size=1000,
		chunk_overlap=100,
		length_function=len,)

	with open(file,'r') as f:
		hp_book = f.read()

	doc_splits = text_splitter.create_documents(texts = [hp_book])

	vectorstore = Chroma.from_documents(
	    documents=doc_splits,
	    collection_name="rag-chroma",
	    embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
	)
	retriever = vectorstore.as_retriever()

	template = """ Você é um especialista em resumir vídeos do YouTube.
		Seu objetivo é criar um resumo de um vídeo.
		Abaixo você encontra a transcrição desse vídeo:
		--------
		{context}
		--------
		O resultado total será um resumo do vídeo, destacando os principais temas por meio de marcadores.
		Você deve ser o mais específico possível, destacando os detalhes apresentados no vídeo.
		Utilize sempre português do Brasil.
		RESUMO:
	"""

	resume_prompt = PromptTemplate.from_template(template)
	resume_chain = (
	    {"context": retriever , "question": RunnablePassthrough()}
	    | resume_prompt
	    | llm
	    | StrOutputParser()
	)
	print(resume_chain.invoke(""))

	rag_template = """Você é um professor especialista em responder perguntas sobre vídeos de tecnologia.
		Seu objetivo é dar respostas concisas e precisas de acordo com o conteúdo do vídeo.
		Abaixo você encontra a transcrição do vídeo em questão:
		--------
		{context}
		--------
		O resultado deve ser uma resposta clara e objetiva, destacando os detalhes apresentados no vídeo.
		Se você não souber a resposta, diga que não sabe ou que o vídeo talvez não aborde esse assunto.
		Utilize sempre português do Brasil. E evite comentar ou agradecer a pergunta feita.

		Pergunta: {question}

		Resposta:
	"""
	rag_prompt = PromptTemplate.from_template(rag_template)
	rag_chain = (
	    {"context": retriever , "question": RunnablePassthrough()}
	    | rag_prompt
	    | llm
	    | StrOutputParser()
	)

	while True:
		query = input("\nMais alguma dúvida? Se não, digite 'tchau' para sair.\n")
		if query=="tchau": break

		print(rag_chain.invoke(query))


if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("USO: python stt.py file DEBUG")
		sys.exit(1)

	file=sys.argv[1]
	DEBUG=int(sys.argv[2])

	if DEBUG==0:
		transcribe(file)
	elif DEBUG==1:
		rag(file)
