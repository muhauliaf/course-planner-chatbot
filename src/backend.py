import dotenv
from langchain.agents import (AgentExecutor, AgentType,
                              create_openai_functions_agent, tool)
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from loader import load_external_links, load_internal_links
from utils import url_to_md

dotenv.load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
external_docembeddings = load_external_links()
internal_docembeddings = load_internal_links()

@tool
def list_courses(quarter: str = None, year: int = 2024) -> str:
    """
    Returns a list of MPCS courses from inputting year and optional quarter:summer,autumn,winter,spring.
    If quarter is not specified, leave it as None and it'll search for entire year.
    The output includes: code, name, link, instructor, location, meeting_times for each course.
    The source link URL is on the last line of the output.
    """
    valid_quarters = ['summer','autumn','winter','spring']
    if quarter: quarter = quarter.lower()
    if quarter not in valid_quarters:
        year = min(year, 2023)
        url = f'https://mpcs-courses.cs.uchicago.edu/{year}-{year-2000+1}/courses'
    elif quarter in ['summer','autumn']:
        url = f'https://mpcs-courses.cs.uchicago.edu/{year}-{year-2000+1}/{quarter}/courses'
    else:
        url = f'https://mpcs-courses.cs.uchicago.edu/{year-1}-{year-2000}/{quarter}/courses'
    text_md = url_to_md(url)
    return text_md
# print(list_courses.invoke({'year':2023,'quarter':'winter'}))

@tool
def course_detail(code: str, quarter: str = 'spring', year: int = 2024) -> str:
    """
    Returns MPCS course detail from course code input, which is in
    "mpcs-<5 digit number>-<1 digit section number>" format
    The output should have these information: name, section, instructor, location, meeting_times,
    fulfills, syllabus, description, content, coursework, textbook, prerequisites, overlapping classes,
    other prequisites, eligible programs.
    The source link URL is on the last line of the output.
    """
    # """Returns MPCS course detail from course code input. The output has these keys: name, section, instructor location, meeting_times, fulfills, description, content, coursework, textbook, prerequisites, overlapping_classes, eligible_programs"""
    course_code = code.replace(' ','-').lower()
    if quarter in ['summer','autumn']:
        url = f'https://mpcs-courses.cs.uchicago.edu/{year}-{year-2000+1}/{quarter}/courses/{course_code}'
    else:
        url = f'https://mpcs-courses.cs.uchicago.edu/{year-1}-{year-2000}/{quarter}/courses/{course_code}'
    text_md = url_to_md(url)
    return text_md

@tool
def search_internal_info(query: str):
    """
    Returns best 3 Langchain documents from searching MPCS website for students,
    with search query as input. The website themselves contain:
    course registration procedures (pre-registration, practicum, non-mpcs, booth class)
    course requests (mpcs and non-mpcs), degree requirements, practicum,
    course catalog and feedback, graduation application, program policies, placement exams,
    computing environment, contact information
    The source link URL is on the metadata of the Langchain document.
    """
    relevant_chunks = internal_docembeddings.similarity_search_with_score(query, k=3)
    chunk_docs = [chunk[0] for chunk in relevant_chunks]
    return chunk_docs

@tool
def search_external_info(query: str):
    """
    Returns best 3 Langchain documents from searching the official MPCS websites
    for external users outside of UChicago, with search query as input. The websites themselves contain:
    about mpcs, various programs (9 course, 12 course, joint, predoctoral),
    admission, faculty, career outcomes, alumni spotlight, contact, and faqs
    The source link URL is on the metadata of the Langchain document.
    """
    relevant_chunks = external_docembeddings.similarity_search_with_score(query, k=3)
    chunk_docs = [chunk[0] for chunk in relevant_chunks]
    return chunk_docs

def create_agent_executor():
    tools = [
        list_courses,
        course_detail,
        search_internal_info,
        search_external_info,
    ]
    # output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    # format_instructions = output_parser.get_format_instructions()
    # print(output_parser)
    # print(format_instructions)
    system_prompt = """
    You are a helpful and talkative course planner for MPCS students. Any answer should have the source link url.
    If you can't find the answer to a question, you truthfully says you don't know.
    The answer should be in markdown and use this format:
    <answer-text>
    Sources: <source-url-1>, <source-url-2>, ...
    """.strip()
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_functions_agent(llm, tools, prompt)
    memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
    return agent_executor

def chat_with_agent_executor(agent_executor: AgentExecutor, input: str) -> str:
    try:
        response = agent_executor.invoke({"input": input})
        return response.get('output',str(response))
    except Exception as e:
        return "I'm sorry. My internal agent encountered error while processing your request."

if __name__ == "__main__":
    agent_executor = create_agent_executor()
    print(chat_with_agent_executor(agent_executor, "who are the two teacher of cloud computing classes?"))
    print(chat_with_agent_executor(agent_executor, "what's the meeting tims for each class?"))
    print(chat_with_agent_executor(agent_executor, "What's the difference between the two classes?"))
    print(chat_with_agent_executor(agent_executor, "Who is Borja Sotomayor? Is he a faculty in MPCS?"))