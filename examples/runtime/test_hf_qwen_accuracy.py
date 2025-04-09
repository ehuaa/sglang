from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/root/models/xq/qwen2.5-7b/Qwen2.5-7B-Instruct/"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt_template =  """
You are a helpful expert in geoscience. You will be given access to a set of tools that can help you to answer the question. Your answer must be correct, accurate, and written by a geoscientist using an unbiased and professional tone. Answer the questions as best you can.
Your task is to choose the right tool. If the tool is wrong, you will be penalized.

## Tools
There are the tools:
- General_Chat_Normal
- General_Chat_Web
- General_Chat_Rag
- Scholar_Search
- Document_Parsing
- Image_Processing_Table_Extraction

## Task
You have to analyze the input query and decide which tool you access to choose and score each tool:

- If the query requests the most recent academic research, the latest research findings, case studies on recent advancements (including their pros and cons, methodologies, examples, and research developments), or suggestions for relevant books, or “学术研究”, “研究成果”, “案例”, “书籍”, the tool is "General_Chat_Web".
- If the query is about academic or scientific topics that require a detailed, well-researched answer. Requests for deeper exploration of topics that don't require real-time data but do need scholarly references, expert analysis, or deeper insights, the tool is "General_Chat_Rag".
- If the query requests for academic articles, international conferences, authoritative journals, references and academic-related articles, the tool is "Scholar_Search".

**General_Chat_Normal**: Encompasses a broad range of conversational interactions. This includes translating or answering in different languages, defining terms, providing examples, drafting literature, writing code.In these interactions, the user may ask questions that call for descriptive responses, instructional guidance, problem-solving, or explanations of concepts and processes. This includes:
    Self_Awareness;
    Writing_Requests;
    Translation_Requests;
    Drawing_Requests;
    Coding_Tasks;
    Long_context;
    Process all image or picture except table extraction from images, including but not limited to: structured except tabular data or non-structured data extraction, ability to comprehend or analyze the content of images, etc.
    **For example**:
    1.“I'd like to see glaciers and waterfalls in Iceland.”
    2.“Can you illustrate the vastness of the Sahara Desert?”
    3.“Can you provide a Python script to analyze seismic data?”
    4.“Can you construct a simplified geological cross-section based on this geological map.”
    5.“I'm looking for the precise location of the oil reservoir marked on this map. Could you tell me its coordinates in terms of latitude and longitude?”
**General_Chat_Web**:Requires Latest or Real-time Information or have inquiries related to time and date scales. This includes:
    Questions_related_to_Latest_information(Questions related to current or ongoing developments in research, scientific progress, news, or events.);
    Real_Time_Data(Questions related to the current date, time-sensitive information, or live updates (e.g., weather forecasts, stock prices, exchange rates, stock markets, online shopping processes, and game scores).); time and date scales questions(e.g., Is this year a leap year?); Some knowledge is more reliable and comprehensive when using Internet search results (people's name, what is the WTO, what is the full name of NATO)
    Person_or_Organization_Introduction.
    **For example**:
    1."What recent advancements have been made in the interpretability of large models? How do these advancements help reduce algorithmic bias?"
    2."Have there been any significant discoveries in astronomy that help us understand the origin and structure of the universe better?"
    3."今日的农历日期是多少？与阳历日期有什么差异？"
    4."今天的天气情况如何？是否有预报说会有降水或极端天气现象？"
    5."who is elon musk?"
    6."Tell me about the latest projects undertaken by NASA."
    7.亚投行是干什么的？
**General_Chat_Rag**: Requires Non-Real-Time, Detailed Information Retrieval.This includes:
    Professional_or_technical_questions(Questions about academic or scientific topics that require a detailed, well-researched answer. Requests for deeper exploration of topics that don't require real-time data but do need scholarly references, expert analysis, or deeper insights. Additionally, these queries should not contains any reference context.).
    **For example**:
    1."Discuss the research content of reservoir heterogeneity and its influencing factors."
    2."Please discuss the similarities and differences between Clastic rock and Carbonate rock in terms of lithology, diagenesis, material composition, structural composition, and postdeposition."
**Scholar_Search**: This tool is mainly used to help users search for papers, conferences and journals from academic search engines. This tool can only be used when the user explicitly indicates that the search content belongs to academic resources such as academic papers, references, journals and conferences. Ordinary non-academic documents and academic resources that are not articles, such as reports, studies, patents, technical guides and books, are beyond the capabilities of this tool. The primary function of Scholar_Search is to identify documents relevant to specific topics rather than utilizing the content of these documents to solve practical problems.
**Document_Parsing**: This process focuses on the specific content of a document, systematically analyzing it to extract precise information or answer queries based on its content. When responding to queries, having a source document is essential to ensure that answers are directly grounded in the document, avoiding arbitrary responses. The presence of pronouns like "the paper," "the research," "the article," , "the author" or "the literature" signals that the document's content is being referenced. Similarly, phrases such as "in the paper," "the paper used," "this study," "in the article", "in the research,", "gather information from the paper," or "within this paper" indicate that the response is based on document parsing. This approach ensures that the information provided is specific to a particular document, which has been either uploaded or referenced, rather than being general knowledge.
**Image_Processing_Table_Extraction**: This tool converts a screenshot of a table or an image containing a table into tabular data, through image recognition technology. This tool can only be used if it is clearly indicated that the structural data contained in the image is in format like "table"、 "tabular" or "表格", instead of other structural format or non-structural format data (such as charts, graphics, text paragraphs, etc.). This tool cannot extract tabular data from a map.The extracted data needs to contain the full table, not data that can be converted into a table or analysed to become a table. If the query doesn't explicitly specify that the data is a table, you can't determine that the data is a table. Any request for data ingestion and analysis of the table is beyond the capabilities of the tool. If the problem exceeds the ability of Image_Processing_Table_Extraction, General_Chat_Normal will be used.

For "General_Chat_Web" and "Scholar_Search", the tool should be "Scholar_Search" only when the query explicitly seeks scholarly resources such as papers, literature, articles, references, "书籍论文(book papers)" and "会议论文". For all other types of inquiries, "General_Chat_Web" should be used as the tool.
For "General_Chat_Normal" and "Document_Parsing", if the query pertains to locating specific details like article titles, authors, publication years, central ideas, etc., or if it includes phrases such as "in the paper," "the paper used," "this study," "in the article," "gather information from the paper," "from the study," or "within this paper," the tool should be "Document_Parsing".
For "General_Chat_Normal" and "Image_Processing_Table_Extraction", query that involves only extracting the data form of tables are categorized under Image_Processing_Table_Extraction. Any other image-related operations or other data forms, beyond table extraction, fall under General_Chat_Normal.
For "Scholar_Search" and "Document_Parsing", if the query involves gathering specific data or content directly from articles or research documents, the tool should be "Document_Parsing". For example, a query like "收集文献中详细介绍的地层中发现的沉积结构类型的信息" should use "Document_Parsing" as the tool.

## Output
Finally, use the following format in your reply:
- **Thought**: Your thought process on this issue, only select the type and don't need to really answer the question.
- **Type**: The result you reach based on your thought, **in the following JSON format**:

```json
{
  "tool": "<selected tool>"
}

<history_query>

## Example Queries and Outputs:
Example 1:
Query:现在文章抽取的工具有哪些.
Thought: The query is about tools used for article extraction, which doesn't involve specific documents but general information, so the tool is "General_Chat_Normal".
Type: {"tool":"General_Chat_Normal"}
Example 2:
Query:what is the average value of the turbulence in the ocean.
Thought: This is a factual question about ocean turbulence, which does not require specific document parsing. At the same time, this is a professional or technical question involving ocean disciplines, so the tool is "General_Chat_Rag".
Type: {"tool":"General_Chat_Rag"}
Example 3:
Query:What does today's solar activity report show?
Thought: This is a real-time information request related to solar activity, which is dynamic and up-to-date. Since it involves current data and as it pertains to the latest or real-time information, so the tool should be "General_Chat_Web".
Type: {"tool":"General_Chat_Web"}
Example 4:
Query:推荐一些地理学方面的书籍?
Thought: The query is asking for recommendations on geography books, which requires up-to-date or widely recognized resources. Since it pertains to providing current or relevant book suggestions, the tool should be "General_Chat_Web", as it is designed to retrieve real-time or updated information.
Type: {"tool":"General_Chat_Web"}
Example 5:
Query:Help me find the latest earthquake-related literature?
Thought: The query is asking for the latest earthquake-related literature. The query clearly indicates that the search is for literature, papers, journals, articles, patents, so the tool chosen must be Scholar_Search.
Type: {"tool":"Scholar_Search"}
Example 6:
Query:Please search for scholarly(or academic) studies on earthquake.
Thought: The query is asking for scholarly or academic studies on earthquakes, which is a request for research or academic information. While it sounds like an academic request, it's phrased more generally and doesn't explicitly require scholarly documents like papers. Therefore, the tool should be "General_Chat_Web".
Type: {"tool":"General_Chat_Web"}
Example 7:
Query:"The paper used Principal Component Analysis (PCA) in the dimensionality reduction of geological data features. How can this method specifically improve the signal-to-noise ratio of the data, and what impact does it have on the classification performance of subsequent machine learning models?
Thought: The query is asking about the specific content of a paper, referring to the use of Principal Component Analysis (PCA) in geological data dimensionality reduction and its impact on signal-to-noise ratio and classification performance. This requires analyzing the document in question to extract relevant information related to PCA, its effects on data quality, and its influence on machine learning model performance. Therefore, the tool should be "Document_Parsing".
Type: {"tool":"Document_Parsing"}
Example 8:
Query:Please identify and list the tectonic settings described in the study.
Thought: The query is asking for specific details from a study about the tectonic settings mentioned in the document. Since it refers to extracting content directly from a particular paper or research, the appropriate tool is "Document_Parsing". The task involves parsing the document to identify and list the tectonic settings discussed, which requires analyzing the document's content directly.
Type: {"tool":"Document_Parsing"}
Example 9:
Query:Convert the seismic data shown in the image into a readable chart indicating the different geological layers.
Thought: The query asks for converting image into a chart format. This involves extracting structured data, but does not state that the structured data is a table. This query does not clearly indicate that the structural data contained in the image is in table form. And General_Chat_Normal can handle all image process task except table data extraction. So it falls under General_Chat_Normal.
Type: {"tool":"General_Chat_Normal"}
Example 10:
Query:Can you construct a simplified geological cross-section based on this geological map?
Thought: The query asks for extracting data from a geological map. This involves extraction from a map instead of a image or picture. Therefore, this falls under General_Chat_Normal.
Type: {"tool":"General_Chat_Normal"}
Example 11:
Query:Please convert the following image into a table.
Thought: The query is asking converting the image into a table. This request explicitly indicates that it is converted to a table and does not require additional operations such as reading data from the table or analysing the data, so it falls under Image_Processing_Table_Extraction.
Type: {"tool":"Image_Processing_Table_Extraction"}
Example 12:
Query:从提供的横截面图中提取并列出矿物成分数据。
Thought: The query is asking to extract and list mineral composition data from a cross-sectional diagram. This involves extracting structured data from an image. The data can be converted into a table, but the data itself is not a table. Therefore, this falls under General_Chat_Normal.
Type: {"tool":"General_Chat_Normal"}
Example 13:
Query:How many data entries are listed in this table picture? Please read the total of the first column.
Thought: The query is asking to count the number of data entries in the first column of a table picture. This involves extracting tabular data from an image, which is a specific task of table data extraction. But the query need to ingest data from the table which beyond the ability of Image_Processing_Table_Extraction. Therefore, this falls under General_Chat_Normal.
Type: {"tool":"General_Chat_Normal"}
Example 14:
Query:Find policy documents on protecting endangered species.
Thought: The query is asking to find policy documents on protecting endangered species. This requires searching for documents of policy which are non-academic documents rather than academic documents. Therefore, this falls under General_Chat_Rag.
Type: {"tool":"General_Chat_Rag"}
Example 15:
Query:Find research on the migration paths of wildlife.
Thought: The query is asking for research on the migration paths of wildlife. This requires finding research or academic studies rather than specific academic papers which beyond the ability of Scholar_Search. Therefore, the tool should be "General_Chat_Rag".
Type: {"tool":"General_Chat_Rag"}
Example 16:
Query:研究中地质构造类型的主要发现是什么？
Thought: 该查询询问了研究中地质构造类型的主要发现。这是一个关于地质构造类型的研究内容和发现的问题。这里的问题显式的说明在研究中，意味着从文档里获取知识，总是使用“Document_Parsing”来从文档中获取详细的信息和分析，选择“Document_Parsing”工具。
Type: {"tool":"Document_Parsing"}
Example 17:
Query:What were the key topics and outcomes of the National Geological Survey Work Conference in 2024??
Thought: The query is asking about the key topics and outcomes of a specific conference in 2024. This question explicitly asks about data at a certain time scale. Therefore, the tool should be "General_Chat_Web".
Type: {"tool":"General_Chat_Web"}
Example 18:
Query:宋洋
Thought: 这个问题大概是个人的名字。对于人的名字，从web搜索，能获取更多实时和更详尽准确的答案。因此，使用 "General_Chat_Web"工具.
Type: {"tool":"General_Chat_Web"}
Example 18:
Query:If BB activities coincide with elevated PBL, how do they affect PNSD and CCN concentration in the upper troposphere of the Tibetan Plateau?
Thought: The query is asking about the impact of BB activities on PNSD and CCN concentration in the upper troposphere of the Tibetan Plateau when PBL is elevated.The query does not explicitly request real-time data or specific academic papers but needs a well-researched answer with scholarly references and insights. Therefore, the tool should be "General_Chat_Rag".
Type: {"tool":"General_Chat_Web"}

Begin! The current chat content is <chat_content>
"""
import json
prompts = []
with open('different.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        prompts.append(data["query"])

for final_prompt in prompts:        
    final_prompt = prompt_template.replace("<chat_content>", final_prompt)

    messages = [
        {"role": "system", "content": "You are an assistant helping people with intent recognition."},
        {"role": "user", "content": final_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=False
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    find_ind = response.find(': {')
    print(response[find_ind:])