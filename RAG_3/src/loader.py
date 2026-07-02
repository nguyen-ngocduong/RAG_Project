from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader

def load_document(directory_path: str):
    """
    Load documents from a directory using UnstructuredFileLoader.
    """
    loader = DirectoryLoader(
        directory_path, 
        glob="**/*.pdf", 
        loader_cls=UnstructuredFileLoader,
        show_progress=True,
        use_multithreading=True)
    documents = loader.load()
    return documents