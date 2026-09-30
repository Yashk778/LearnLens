from langchain_text_splitters import RecursiveCharacterTextSplitter 

def chunk_text(text:str,chunk_size:int = 1000, chunk_overlap:int=150 ) -> list[str]:

    if not text:
        raise ValueError('Text to chunk cannot be empty.')

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap=chunk_overlap

    )

    return splitter.split_text(text)