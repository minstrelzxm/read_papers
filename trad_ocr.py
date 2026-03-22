from PyPDF2 import PdfReader
# The comparison between traditional package PyPDF2 here and new LLM-based OCR model is huge.
# PyPDF2 cannot identify the structure of the paper, and cannot properly extract the formula and images.
# It is just a simple text extraction tool.

reader = PdfReader("original_papers/A_Hierarchy_of_Graphical_Models_for_Counterfactual_Inferences_uJinTwXfbs.pdf")
number_of_pages = len(reader.pages)
print(number_of_pages)

for i in range(number_of_pages):
    page = reader.pages[i]
    text = page.extract_text()
    print(f"Page {i}: {text}")

    # append the text into a file
    with open(f"extracted_papers/A_Hierarchy_of_Graphical_Models_for_Counterfactual_Inferences_uJinTwXfbs/full_extracted_trad.md", "a") as f:
        f.write(text)
        f.write("\n")