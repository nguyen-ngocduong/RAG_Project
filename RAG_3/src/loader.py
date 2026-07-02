import pymupdf

def load_pdf(pdf_path):
    """
    -Load theo tung trang cua pdf
    """
    doc = pymupdf.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(
            page.get_text()
        )
    return pages