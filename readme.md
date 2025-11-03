# AI Agent for CE 331 Lab Report Automation

Applicant Details
Name: Yash Pathak
Roll No.: 231188
University: Indian Institute of Technology Kanpur (IIT Kanpur)
Department: Civil Engineering

## 1. Project Objective

This project automates the manual, time-consuming task of writing university lab reports for the
course CE331: Principles of Geoinformatics.

The goal was to build a sophisticated AI agent that can reason, plan, and execute a multi-step process
to generate a complete, formatted, and accurate lab report. The agent functions as an AI collaborator:
the user provides specific inputs (lab title, key concepts, and raw experiment data), and the agent
handles all the research, writing, data formatting, image sourcing, and file conversion, delivering a
finished PDF.

## 2. LLM Interaction Logs

As required by the assignment, these logs document the use of LLMs during the development and
debugging of this agent.
Google Gemini Interaction Log:https://gemini.google.com/share/ 4 a 366 b 324623
OpenAI ChatGPT Interaction Log: https://chatgpt.com/share/6907c6ee-530c-8009-a940-e05e3cd21b03
## 3. How to Run the Agent (Source Code)

The primary source code is the final_agent_notebook.ipynb file. It is designed to be run in a cloud
GPU environment (like Google Colab) to avoid local dependency conflicts and to leverage the
necessary hardware for the 7B-parameter model.

Prerequisites
A Google Colab account.
A T4 GPU runtime.
Your two Google API keys (API Key & CSE ID) for Agent 2 (Image Search).
The course_materials.zip file (the RAG knowledge base).
The mistral-lab-report-trainer.zip file (the fine-tuned LoRA adapter).

Step-by-Step Instructions

1. Open in Google Colab:
    Create a new Colab notebook.
    Click File > Upload notebook... and select final_agent_notebook.ipynb.


2. Set the Runtime:
    Click Runtime > Change runtime type.
    Select T4 GPU from the "Hardware accelerator" dropdown and click Save.
3. Upload Your Project Files:
    In the file browser on the left-hand side, click the "Upload" icon.
    Upload your zipped knowledge base: course_materials.zip
    Upload your zipped fine-tuned model: mistral-lab-report-trainer.zip
4. Run the Cells in Order:
    Cell 1: Run the installer cell. This will install all required libraries (e.g., transformers, peft,
       chromadb-client).
    Cell 2: Run the unzip cell. This will extract your files into the Colab environment.
    Cell 3: Run the ChromaDB cell. This will create the RAG knowledge base from your
       course_materials PDFs.
5. Configure and Run the Final Agent (Cell 4):
    Go to Cell 4.
    Find the --- 0. API KEYS --- section.
    Paste your API keys into the GOOGLE_API_KEY and GOOGLE_CSE_ID variables.
    Run Cell 4. The script will load the 7B model (1-2 min), load the ChromaDB (5 sec), and then
    automatically run the full 4-agent pipeline for the pre-set "Lab 4 Levelling" example.
    The generation will be fast (30-90 seconds).
6. Get Your Output:
    The generated Markdown text will be printed to the output.
    A final PDF file (e.g., Lab_Report_Lab_4_Levelling_...pdf) will be created. You can
    download this from the file browser on the left.

How to Generate a _New_ Lab Report
To generate your own report, simply modify the variables in Cell 4 under the --- PART 3: RUN THE
4-AGENT SYSTEM --- section.

```
# --- PART 3: RUN THE 4-AGENT SYSTEM ---
# 1. Change the Lab Title
lab_title = "Lab 5 Close Traverse"
# 2. Change the Key Concepts for Agent 1 (RAG)
# This is the most important input for getting good context.
key_concepts = "Bowditch's Rule, angular misclosure, linear misclosure, traverse"
# 3. Paste your new experiment data
# The fine-tuned model is trained to find and format this data.
exp_data = """
| Line | WCB | Length(m) | Latitude | Departure |
|---|---|---|---|---|
| AB | 3.5° | 53.685 | 53.58 | 3.27 |
| BC | 249.36° | 38.739 | -13.66 | -36.24 |
```

```
...
"""
# 4. Re-run Cell 4
# The agent will now generate a new report for "Lab 5 Close Traverse"
```
## 4. AI Agent Architecture

This prototype is a multi-agent system that fulfills the "Multi-agent collaboration" and "RAG" bonus
requirements. It uses a "human-in-the-loop" pipeline where a central script coordinates four
specialized agents to achieve the final goal.

Interaction Flow

1. User Input: The user provides the lab_title, key_concepts, and exp_data.
2. Agent 2 (Image Search) is triggered, using the lab_title to find a relevant image URL via the
    Google Search API.
3. Agent 1 (Knowledge - RAG) is triggered, using the key_concepts to perform a semantic search
    on the ChromaDB vector store. It retrieves the most relevant text chunks from the course books.
4. Agent 3 (The Writer) receives a single, large prompt containing all inputs: the user's task, the
    new exp_data, the image_url (from Agent 2), and the book_context (from Agent 1).
5. Agent 3 (The Writer) syntesizes all this information into a single, cohesive Markdown report. Its
    fine-tuning ensures the structure is correct and the exp_data is formatted into tables.
6. Agent 4 (The Publisher) receives the final Markdown text and converts it into a .pdf file.

Component Deep-Dive
Agent 1: The "Knowledge Agent" (RAG)
Model: sentence-transformers/all-MiniLM-L6-v
Component: ChromaDB
Function: Acts as the agent's long-term memory. It retrieves factual, academic context
(definitions, formulas) from the course textbooks to prevent the Writer agent from
hallucinating.
Agent 2: The "Image Agent" (Tool Use)
Model: Google Custom Search API
Component: googleapiclient
Function: Acts as a specialized tool to find external information (images) that is not present in
the knowledge base, fulfilling a common requirement for lab reports.
Agent 3: The "Writer" (Fine-Tuned LLM)
Base Model: mistralai/Mistral-7B-Instruct-v0.
Fine-Tuned Adapter: checkpoint-6 (A PEFT/LoRA adapter trained for 3 epochs)
Function: This is the core "brain" of the operation. It takes all the disparate pieces of context
and synthesizes them into a final, human-readable report.


```
Agent 4: The "Publisher" (PDF Converter)
Component: markdown-pdf library
Function: This agent completes the automation pipeline by converting the structured text
output from Agent 3 into a final, shareable .pdf document.
```
## 5. Justification for Fine-Tuning Agent 3

The assignment's core requirement was to fine-tune a model. Agent 3 (The Writer) was the explicit
target for this, as it provides the most value.

A base, general-purpose model like Mistral-7B is not sufficient for this task. We used LoRA (Low-Rank
Adaptation) to teach it two specific skills:

1. Adapted Style: The most obvious failure of a base model is that it does not know the specific,
    rigid structure of a CE331 lab report. Our fine-tuning on 5 example reports taught the model to
    reliably generate the correct Markdown structure every time: ## Objective, ##
    Equipment, ## Methodology, ## Results and Discussion, and ## Conclusions.
2. Task Specialization (Data Transformation): This is the most important skill. The agent must be
    able to take raw, messy data (pasted by the user) and re-format it into clean Markdown
    tables. Our finetune_data.jsonl file specifically trained the model on this "data-in, table-out"
    transformation. This is a highly specialized task that a base model cannot perform reliably.


