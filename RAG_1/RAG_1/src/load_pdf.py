from pypdf import PdfReader
def load_pdf_from_page(pdf_path, start_page=14):
    " start_page=16 vì PdfReader đánh số từ 0 PDF page 14 => index 14 "

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages[start_page:]:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text
