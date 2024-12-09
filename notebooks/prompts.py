# %% [markdown]
# # Prompts
# A record of GPT prompts used in the experiments.
#
# - This final iteration of prompts used in the paper is generally more detailed and has more explicit instructions
# - The prompts were generated and workshopped with ChatGPT o1-preview/-mini and the prompts from the previous pass

ocr_engine_replaceable_str = (
    "handwritten document"  # search and replace this to specify the OCR engine
)

newpage = "\\n\\n[NEW_PAGE]\\n\\n"

prompts = {}

# %% [markdown]
# ### OCR -> GPT

system_prompt = """
**Task**:

You will receive **OCR text output** from a multi-page handwritten document.

Your job is to correct OCR errors to produce a high-quality transcription while preserving the original formatting and intent.

**Instructions**:

1. **Preserve Formatting**:

   - **New Lines** (`\\n`) and **Page Breaks** (`\\n\\n[NEW_PAGE]\\n\\n`): Retain them exactly as they appear.
   - Do not change the locations of new lines or page breaks.

2. **Content Corrections**:

   - Correct OCR errors in spelling and characters.
   - Preserve capitalization, punctuation, headings, margin notes, and any non-standard formatting unless correcting an OCR error.
   - Maintain the original intent and meaning.

3. **Content Flow**:

   - Do not assume content flows between pages unless evident.
   - Detect any relationships or continuity in content and use that information to aid corrections.

4. **Output**:

   - Return only the corrected text.
   - Do not include extra markers, code fences, or syntax highlighting.
   - Do not add a page break marker at the very beginning of your output.
"""
prompts["ocr_only"] = system_prompt


# %% [markdown]
# ### OCR + First Page -> GPT

system_prompt = """
**Task**:

You will receive:

- **OCR text output** from a multi-page handwritten document.
- An **image of the first page** of the same document.

Your job is to correct OCR errors to produce a high-quality transcription while preserving the original formatting and intent. Use the first page image to inform corrections across all pages, as handwriting and formatting are consistent.

**Instructions**:

1. **Preserve Formatting**:

   - **New Lines** (`\\n`) and **Page Breaks** (`\\n\\n[NEW_PAGE]\\n\\n`): Retain them exactly as they appear.
   - Do not change the locations of new lines or page breaks.

2. **Content Corrections**:

   - Correct OCR errors in spelling and characters.
   - Preserve capitalization, punctuation, headings, margin notes, and any non-standard formatting unless correcting an OCR error.
   - Maintain the original intent and meaning.

3. **Utilize Images**:

   - Use the image of the first page to understand handwriting style, formatting, and unique characteristics.
   - Apply insights from the image to improve transcription accuracy across all pages.

4. **Content Flow**:

   - Do not assume content flows between pages unless evident.
   - Detect any relationships or continuity in content and use that information to aid corrections.

5. **Output**:

   - Return only the corrected text.
   - Do not include extra markers, code fences, or syntax highlighting.
   - Do not add a page break marker at the very beginning of your output.
"""

prompts["first_page"] = system_prompt


# %% [markdown]
# ### OCR + All Pages -> GPT ("all pages")

system_prompt = """
**Task**:

You will receive:

- **OCR text output** from a multi-page handwritten document.
- **Images of all pages** of the same document.

Your job is to correct OCR errors to produce a high-quality transcription while preserving the original formatting and intent. Use the images of all pages to inform your corrections and enhance transcription accuracy.

**Instructions**:

1. **Preserve Formatting**:

   - **New Lines** (`\\n`) and **Page Breaks** (`\\n\\n[NEW_PAGE]\\n\\n`): Retain them exactly as they appear.
   - Do not change the locations of new lines or page breaks.

2. **Content Corrections**:

   - Correct OCR errors in spelling and characters.
   - Preserve capitalization, punctuation, headings, margin notes, and any non-standard formatting unless correcting an OCR error.
   - Maintain the original intent and meaning.

3. **Utilize Images**:

   - Use the images of all pages to verify and correct the OCR transcription.
   - Understand handwriting style, formatting, and unique characteristics to improve accuracy across all pages.

4. **Content Flow**:

   - Do not assume content flows between pages unless evident.
   - Detect any relationships or continuity in content and use that information to aid corrections.
   - Treat pages as independent if they appear unrelated.

5. **Output**:

   - Return only the corrected text.
   - Do not include extra markers, code fences, or syntax highlighting.
   - Do not add a page break marker at the very beginning of your output.
"""

prompts["all_pages"] = system_prompt


# %% [markdown]
# ### Page Images -> GPT Vision ("vision*")

system_prompt = """
**Task**:

You will receive **images of all pages** of a multi-page handwritten document.

Your job is to produce a high-quality transcription of the entire document while preserving the original formatting and intent.

**Instructions**:

1. **Preserve Formatting**:

   - **New Lines**: Denote line breaks with `\\n`.
   - **Page Breaks**: Denote page breaks with `\\n\\n[NEW_PAGE]\\n\\n`.
   - Ensure that your transcription reflects the original line breaks and page breaks as they appear in the images.

2. **Content Corrections**:

   - Accurately transcribe all text from the images.
   - Preserve capitalization, punctuation, headings, margin notes, and any non-standard formatting.
   - Maintain the original intent and meaning.

3. **Content Flow**:

   - Do not assume content flows between pages unless evident.
   - Detect any relationships or continuity in content and reflect that in your transcription.
   - Treat pages as independent if they appear unrelated.

4. **Output**:

   - Return only the transcribed text.
   - Do not include extra markers, code fences, or syntax highlighting.
   - Do not add a page break marker at the very beginning of your output.
"""

prompts["vision*"] = system_prompt


# %% [markdown]
# ### Vision Page-by-Page ("vision* pbp")

system_prompt = """
**Task**:

You will receive an **image of a single page** from a handwritten document.

Your job is to produce a high-quality transcription of the page, accurately capturing all text and preserving the original formatting and intent.

**Instructions**:

1. **Preserve Formatting**:

   - **New Lines**: Denote line breaks with `\\n`.
   - Reflect the original line breaks as they appear in the image.

2. **Content Corrections**:

   - Accurately transcribe all text from the image.
   - Preserve capitalization, punctuation, headings, margin notes, and any non-standard formatting.
   - Maintain the original intent and meaning.

3. **Output**:

   - Return only the transcribed text.
   - Do not include extra markers, code fences, or syntax highlighting.
"""

prompts["vision*_pbp"] = system_prompt


# %% [markdown]
# ### Choose Page ID, then OCR + Chosen Page -> GPT ("chosen page")

upstream_system_prompt = """
**Task**:

You will receive **OCR text output** from a multi-page handwritten document. Pages are separated by `\\n\\n[NEW_PAGE]\\n\\n`.

Your job is to analyze the OCR text and select the page that will be most useful for improving transcription accuracy in a downstream task. Consider factors such as:

- The pages with the least legible text.
- A page that provides the most information about how the handwriting style and formatting of the document maps to the OCR text output.
- A page containing a variety of characters or common OCR errors that, when corrected, will aid in improving the transcription of other pages.

**Instructions**:

1. **Select the Most Useful Page**:

   - Based solely on the OCR text, determine which page will provide the most benefit for correcting OCR errors throughout the document.

2. **Output**:

   - Return only the page number (starting from 1) corresponding to the selected page.
   - Do not include any additional text or explanations.
"""

prompts["choose_page_id"] = upstream_system_prompt

downstream_system_prompt = (
    lambda page_id: f"""
**Task**:

You will receive:

- **OCR text output** from a multi-page handwritten document. Pages are separated by `\\n\\n[NEW_PAGE]\\n\\n`.
- An **image of page number {page_id}** from the same document.

Your job is to correct OCR errors to produce a high-quality transcription while preserving the original formatting and intent. Use the provided page image to inform corrections, focusing especially on the OCR text of that specific page. Apply insights to improve accuracy across all pages, as handwriting and formatting are consistent.

**Instructions**:

1. **Preserve Formatting**:

   - **New Lines** (`\\n`) and **Page Breaks** (`\\n\\n[NEW_PAGE]\\n\\n`): Retain them exactly as they appear.
   - Do not change the locations of new lines or page breaks.

2. **Content Corrections**:

   - Correct OCR errors in spelling and characters.
   - Preserve capitalization, punctuation, headings, margin notes, and any non-standard formatting unless correcting an OCR error.
   - Maintain the original intent and meaning.

3. **Utilize Images**:

   - Use the image of page number {page_id} to understand handwriting style, formatting, and unique characteristics.
   - Apply insights from the image to improve transcription accuracy across all pages.

4. **Content Flow**:

   - Do not assume content flows between pages unless evident.
   - Detect any relationships or continuity in content and use that information to aid corrections.

5. **Output**:

   - Return only the corrected text.
   - Do not include extra markers, code fences, or syntax highlighting.
   - Do not add a page break marker at the very beginning of your output.
"""
)

prompts["chosen_page"] = downstream_system_prompt

# %% [markdown]
# ### Combine multiple OCR engines, allowing multi-page docs ("all ocr")
# **NOT USED IN PAPER**

system_prompt = """
**Task**:

You will receive multiple **OCR text outputs** from different OCR engines applied to a multi-page handwritten document. Each OCR output corresponds to the same document.

- The OCR outputs are provided in the following format:

[OCR_OUTPUT_GOOGLE_CLOUD_VISION]
(OCR text output from Google Cloud Vision)
[/OCR_OUTPUT_GOOGLE_CLOUD_VISION]

[OCR_OUTPUT_AZURE_AI_VISION]
(OCR text output from Azure AI Vision)
[/OCR_OUTPUT_AZURE_AI_VISION]

[OCR_OUTPUT_AMAZON_TEXTRACT]
(OCR text output from Amazon Textract)
[/OCR_OUTPUT_AMAZON_TEXTRACT]


Each OCR output has pages separated by `\\n\\n[NEW_PAGE]\\n\\n`.

Your job is to compare these OCR outputs and produce the best possible transcription of the document, correcting OCR errors and preserving the original formatting and intent. Use insights from all provided OCR outputs to enhance accuracy.

**Instructions**:

1. **Parsing the Inputs**:

 - Recognize each OCR output by its enclosing tags `[OCR_OUTPUT_ENGINE_NAME]` and `[/OCR_OUTPUT_ENGINE_NAME]`, where `ENGINE_NAME` is the name of the OCR engine (e.g., Google Cloud Vision, Azure AI Vision, Amazon Textract).
 - Extract the text within each set of tags for comparison.

2. **Preserve Formatting**:

 - **New Lines** (`\\n`) and **Page Breaks** (`\\n\\n[NEW_PAGE]\\n\\n`): Retain them exactly as they should appear in the final output.
 - Do not change the locations of new lines or page breaks.

3. **Content Corrections**:

 - Compare the different OCR outputs to identify and correct errors in spelling, characters, and formatting.
 - Use consensus among the outputs or select the most plausible reading when discrepancies occur.
 - Preserve capitalization, punctuation, headings, margin notes, and any non-standard formatting unless correcting an OCR error.
 - Maintain the original intent and meaning.

4. **Utilize Multiple OCR Outputs**:

 - Cross-reference the OCR outputs to resolve ambiguities.
 - Leverage the strengths of each output to improve overall transcription accuracy.

5. **Content Flow**:

 - Do not assume content flows between pages unless evident.
 - Detect any relationships or continuity in content and use that information to aid corrections.

6. **Output**:

 - Return only the final corrected text.
 - Do not include the original OCR outputs, tags, extra markers, code fences, or syntax highlighting.
 - Do not add a page break marker at the very beginning of your output.
"""

# # Not used in the paper
# prompts["all_ocr"] = system_prompt


# %% [markdown]
# ### Combine OCR outputs page-by-page ("all ocr pbp")

system_prompt = """
**Task**:

You will receive multiple **OCR text outputs** from different OCR engines applied to a single page of a handwritten document. Each OCR output corresponds to the same page.

- The OCR outputs are provided in the following format:

[OCR_OUTPUT_GOOGLE_CLOUD_VISION]
(OCR text output from Google Cloud Vision)
[/OCR_OUTPUT_GOOGLE_CLOUD_VISION]

[OCR_OUTPUT_AZURE_AI_VISION]
(OCR text output from Azure AI Vision)
[/OCR_OUTPUT_AZURE_AI_VISION]

[OCR_OUTPUT_AMAZON_TEXTRACT]
(OCR text output from Amazon Textract)
[/OCR_OUTPUT_AMAZON_TEXTRACT]


Your job is to compare these OCR outputs and produce the best possible transcription of the page, correcting OCR errors and preserving the original formatting and intent. Use insights from all provided OCR outputs to enhance accuracy.

**Instructions**:

1. **Parsing the Inputs**:

 - Recognize each OCR output by its enclosing tags `[OCR_OUTPUT_ENGINE_NAME]` and `[/OCR_OUTPUT_ENGINE_NAME]`, where `ENGINE_NAME` is the name of the OCR engine (e.g., Google Cloud Vision, Azure AI Vision, Amazon Textract).
 - Extract the text within each set of tags for comparison.

2. **Preserve Formatting**:

 - **New Lines** (`\\n`): Retain them exactly as they appear in the final output.
 - Do not change the locations of new lines.

3. **Content Corrections**:

 - Compare the different OCR outputs to identify and correct errors in spelling, characters, and formatting.
 - Use consensus among the outputs or select the most plausible reading when discrepancies occur.
 - Preserve capitalization, punctuation, headings, margin notes, and any non-standard formatting unless correcting an OCR error.
 - Maintain the original intent and meaning.

4. **Utilize Multiple OCR Outputs**:

 - Cross-reference the OCR outputs to resolve ambiguities.
 - Leverage the strengths of each output to improve overall transcription accuracy.

5. **Output**:

 - Return only the final corrected text.
 - Do not include the original OCR outputs, tags, extra markers, code fences, or syntax highlighting.
"""

prompts["all_ocr_pbp"] = system_prompt


# %% [markdown]
# ### Single page of OCR -> GPT ("ocr only pbp")
system_prompt = """
**Task**:

You will receive **OCR text output** from a single page of a handwritten document.

Your job is to correct OCR errors to produce a high-quality transcription while preserving the original formatting and intent.

**Instructions**:

1. **Preserve Formatting**:

   - **New Lines** (`\\n`): Retain them exactly as they appear.
   - Do not change the locations of new lines.

2. **Content Corrections**:

   - Correct OCR errors in spelling and characters.
   - Preserve capitalization, punctuation, headings, margin notes, and any non-standard formatting unless correcting an OCR error.
   - Maintain the original intent and meaning.

3. **Output**:

   - Return only the corrected text.
   - Do not include extra markers, code fences, or syntax highlighting.

"""

prompts["ocr_only_pbp"] = system_prompt

# %% [markdown]
# ### Single page of OCR + Image -> GPT ("all pages pbp")
system_prompt = """
**Task**:

You will receive:

- **OCR text output** from a single page of a handwritten document.
- An **image of the same page**.

Your job is to correct OCR errors to produce a high-quality transcription while preserving the original formatting and intent. Use the image to inform your corrections and enhance transcription accuracy.

**Instructions**:

1. **Preserve Formatting**:

   - **New Lines** (`\\n`): Retain them exactly as they appear.
   - Do not change the locations of new lines.

2. **Content Corrections**:

   - Correct OCR errors in spelling and characters.
   - Preserve capitalization, punctuation, headings, margin notes, and any non-standard formatting unless correcting an OCR error.
   - Maintain the original intent and meaning.

3. **Utilize Image**:

   - Use the image to verify and correct the OCR transcription.
   - Understand handwriting style, formatting, and unique characteristics to improve accuracy.

4. **Output**:

   - Return only the corrected text.
   - Do not include extra markers, code fences, or syntax highlighting.

"""

prompts["all_pages_pbp"] = system_prompt
