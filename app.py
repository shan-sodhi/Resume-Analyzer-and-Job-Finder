# app.py
# Resume Analyzer + LinkedIn Scraper (Pure Gemini, Simple Prompt Mode)
# - Simple mode: send (truncated) full resume text to Gemini for each task.
# - No LangChain, no FAISS.

import os
import time
import math
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader

# Gemini SDK (google-generativeai)
import google.generativeai as genai

# Selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException

# webdriver-manager fallback
from webdriver_manager.chrome import ChromeDriverManager

import warnings
warnings.filterwarnings("ignore")


# ----------------------- Config & Helpers -----------------------
MAX_RESUME_CHARS = 30000  # simple-mode truncation safeguard
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"  # use one of your available models


def streamlit_rerun():
    """Compatible rerun for old/new Streamlit versions."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def configure_gemini(api_key: str):
    """Configure google.generativeai safely."""
    if not api_key:
        raise ValueError("Gemini (Google) API key required.")
    genai.configure(api_key=api_key)


def try_gemini_call(prompt: str, model: str = DEFAULT_GEMINI_MODEL, max_output_tokens: int = 1024):
    """
    Simple Gemini call compatible with google-generativeai 0.8.5:
    uses GenerativeModel.generate_content only.
    """
    model_obj = genai.GenerativeModel(model)
    resp = model_obj.generate_content(prompt)
    if hasattr(resp, "text") and resp.text:
        return resp.text
    return str(resp)


def safe_call_gemini(prompt: str, api_key: str, model: str = DEFAULT_GEMINI_MODEL):
    """Configure then call Gemini, returning string result."""
    configure_gemini(api_key)
    return try_gemini_call(prompt, model=model)


def extract_text_from_pdf(file_obj) -> str:
    """Return combined text from a PDF file-like object."""
    reader = PdfReader(file_obj)
    text_parts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        text_parts.append(t)
    return "\n".join(text_parts).strip()


def prepare_simple_prompt(task: str, resume_text: str) -> str:
    """Task-specific wrapper that produces a final prompt to send to Gemini."""
    header = (
        "You are an expert resume analyst and ATS optimization specialist. "
        "Use ONLY the resume content below to answer the user's request. "
        "Do not hallucinate. If information is missing, say 'Not mentioned in resume.'\n\n"
        "Resume content:\n"
    )
    if len(resume_text) > MAX_RESUME_CHARS:
        resume_text = resume_text[:MAX_RESUME_CHARS] + "\n\n[TRUNCATED - resume too long]"
    footer = f"\n\nTask: {task}\n\nProvide a concise, well-structured response."
    return header + resume_text + footer


# ----------------------- Resume Analyzer UI functions -----------------------
class ResumeAnalyzer:
    @staticmethod
    def resume_summary_ui(gemini_api_key):
        with st.form(key="summary_form"):
            add_vertical_space(1)
            pdf = st.file_uploader("Upload Your Resume (PDF)", type="pdf")
            add_vertical_space(1)
            model_input = st.text_input("Gemini model (optional)", value=DEFAULT_GEMINI_MODEL, key="summary_model")
            submit = st.form_submit_button("Get Summary")

        api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if submit:
            if pdf is None:
                st.warning("Please upload a PDF resume.")
                return
            if not api_key:
                st.warning("Please provide a Gemini API key in the sidebar or environment.")
                return

            with st.spinner("Extracting and summarizing..."):
                text = extract_text_from_pdf(pdf)
                if not text:
                    st.error("No text found inside PDF.")
                    return
                prompt = prepare_simple_prompt(
                    "Summarize the resume: include 6 bullets for skills/achievements, 1-line career headline, and 2-sentence conclusion.",
                    text,
                )
                try:
                    out = safe_call_gemini(prompt, api_key, model=model_input or DEFAULT_GEMINI_MODEL)
                except Exception as e:
                    st.error(f"Gemini call failed: {e}")
                    return

            st.markdown("<h4 style='color:orange;'>Summary</h4>", unsafe_allow_html=True)
            st.write(out)

    @staticmethod
    def resume_strength_ui(gemini_api_key):
        with st.form(key="strength_form"):
            add_vertical_space(1)
            pdf = st.file_uploader("Upload Your Resume (PDF)", type="pdf", key="strength_pdf")
            add_vertical_space(1)
            model_input = st.text_input("Gemini model (optional)", value=DEFAULT_GEMINI_MODEL, key="strength_model")
            submit = st.form_submit_button("Analyze Strengths")

        api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if submit:
            if pdf is None:
                st.warning("Please upload a PDF resume.")
                return
            if not api_key:
                st.warning("Please provide a Gemini API key.")
                return

            with st.spinner("Analyzing strengths..."):
                text = extract_text_from_pdf(pdf)
                if not text:
                    st.error("No text found inside PDF.")
                    return
                prompt = prepare_simple_prompt(
                    "List 5 main strengths found in the resume, explain briefly why each is a strength, and give 2 tips to highlight it on LinkedIn.",
                    text,
                )
                try:
                    out = safe_call_gemini(prompt, api_key, model=model_input or DEFAULT_GEMINI_MODEL)
                except Exception as e:
                    st.error(f"Gemini call failed: {e}")
                    return

            st.markdown("<h4 style='color:orange;'>Strengths</h4>", unsafe_allow_html=True)
            st.write(out)

    @staticmethod
    def resume_weakness_ui(gemini_api_key):
        with st.form(key="weakness_form"):
            add_vertical_space(1)
            pdf = st.file_uploader("Upload Your Resume (PDF)", type="pdf", key="weak_pdf")
            add_vertical_space(1)
            model_input = st.text_input("Gemini model (optional)", value=DEFAULT_GEMINI_MODEL, key="weak_model")
            submit = st.form_submit_button("Analyze Weaknesses")

        api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if submit:
            if pdf is None:
                st.warning("Please upload a PDF resume.")
                return
            if not api_key:
                st.warning("Please provide a Gemini API key.")
                return

            with st.spinner("Analyzing weaknesses..."):
                text = extract_text_from_pdf(pdf)
                if not text:
                    st.error("No text found inside PDF.")
                    return
                prompt = prepare_simple_prompt(
                    "List 5 weaknesses or gaps in this resume and provide concrete, actionable suggestions to fix each (checklist style).",
                    text,
                )
                try:
                    out = safe_call_gemini(prompt, api_key, model=model_input or DEFAULT_GEMINI_MODEL)
                except Exception as e:
                    st.error(f"Gemini call failed: {e}")
                    return

            st.markdown("<h4 style='color:orange;'>Weaknesses & Suggestions</h4>", unsafe_allow_html=True)
            st.write(out)

    @staticmethod
    def job_title_suggestion_ui(gemini_api_key):
        with st.form(key="jobtitle_form"):
            add_vertical_space(1)
            pdf = st.file_uploader("Upload Your Resume (PDF)", type="pdf", key="jobtitles_pdf")
            add_vertical_space(1)
            model_input = st.text_input("Gemini model (optional)", value=DEFAULT_GEMINI_MODEL, key="jobtitles_model")
            submit = st.form_submit_button("Suggest Job Titles")

        api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if submit:
            if pdf is None:
                st.warning("Please upload a PDF resume.")
                return
            if not api_key:
                st.warning("Please provide a Gemini API key.")
                return

            with st.spinner("Generating job title suggestions..."):
                text = extract_text_from_pdf(pdf)
                if not text:
                    st.error("No text found inside PDF.")
                    return
                prompt = prepare_simple_prompt(
                    "Suggest 10 LinkedIn-style job titles suitable for this candidate. For each title provide a one-line rationale.",
                    text,
                )
                try:
                    out = safe_call_gemini(prompt, api_key, model=model_input or DEFAULT_GEMINI_MODEL)
                except Exception as e:
                    st.error(f"Gemini call failed: {e}")
                    return

            st.markdown("<h4 style='color:orange;'>Job Titles</h4>", unsafe_allow_html=True)
            st.write(out)

    @staticmethod
    def ats_score_ui(gemini_api_key):
        with st.form(key="ats_form"):
            add_vertical_space(1)
            pdf = st.file_uploader("Upload Your Resume (PDF)", type="pdf", key="ats_pdf")
            add_vertical_space(1)
            job_desc = st.text_area(
                "Paste Job Description (optional but recommended for ATS match score)",
                height=200,
                key="ats_jd"
            )
            model_input = st.text_input("Gemini model (optional)", value=DEFAULT_GEMINI_MODEL, key="ats_model")
            submit = st.form_submit_button("Calculate ATS Score")

        api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if submit:
            if pdf is None:
                st.warning("Please upload a PDF resume.")
                return
            if not api_key:
                st.warning("Please provide a Gemini API key.")
                return

            with st.spinner("Calculating ATS score..."):
                text = extract_text_from_pdf(pdf)
                if not text:
                    st.error("No text found inside PDF.")
                    return

                if job_desc.strip():
                    task = (
                        "You are an Applicant Tracking System (ATS) expert. Compare the resume to the job description below.\n\n"
                        f"Job Description:\n{job_desc}\n\n"
                        "Tasks:\n"
                        "1. Give an overall ATS match score between 0 and 100.\n"
                        "2. List important keywords/skills that are MATCHED.\n"
                        "3. List important keywords/skills that are MISSING.\n"
                        "4. Briefly explain why you gave that score.\n"
                        "5. Provide 5–10 very specific edits the candidate should make to improve ATS score for this job.\n\n"
                        "Format the answer clearly with headings:\n"
                        "- ATS Score: <score>/100\n"
                        "- Matched Keywords\n"
                        "- Missing Keywords\n"
                        "- Explanation\n"
                        "- Improvement Suggestions"
                    )
                else:
                    task = (
                        "You are an Applicant Tracking System (ATS) expert. Evaluate this resume for general ATS friendliness "
                        "(for typical roles that match the profile).\n\n"
                        "Tasks:\n"
                        "1. Give an overall ATS score between 0 and 100.\n"
                        "2. List strengths from an ATS perspective (keywords, structure, clarity).\n"
                        "3. List weaknesses from an ATS perspective.\n"
                        "4. Give 5–10 actionable suggestions to improve the ATS score.\n\n"
                        "Format the answer clearly with headings and show score as: ATS Score: <score>/100."
                    )

                prompt = prepare_simple_prompt(task, text)

                try:
                    out = safe_call_gemini(prompt, api_key, model=model_input or DEFAULT_GEMINI_MODEL)
                except Exception as e:
                    st.error(f"Gemini call failed: {e}")
                    return

            st.markdown("<h4 style='color:orange;'>ATS Score & Analysis</h4>", unsafe_allow_html=True)
            st.write(out)


# ----------------------- Resume Chat (conversational Q&A over resume) -----------------------
class ResumeChat:
    @staticmethod
    def chat_ui(gemini_api_key):
        api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if not api_key:
            st.warning("Please provide a Gemini API key in the sidebar or environment.")
            return

        # Initialise session state for chat
        if "resume_text_chat" not in st.session_state:
            st.session_state.resume_text_chat = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "chat_model" not in st.session_state:
            st.session_state.chat_model = DEFAULT_GEMINI_MODEL

        # If resume not yet uploaded for chat: show uploader form
        if st.session_state.resume_text_chat is None:
            with st.form("chat_resume_form"):
                add_vertical_space(1)
                pdf = st.file_uploader("Upload your resume (PDF) to start chat", type="pdf", key="chat_pdf")
                add_vertical_space(1)
                model_input = st.text_input(
                    "Gemini model (optional)",
                    value=DEFAULT_GEMINI_MODEL,
                    key="chat_model_input",
                )
                submit = st.form_submit_button("Start Chat")

            if submit:
                if pdf is None:
                    st.warning("Please upload a PDF resume.")
                    return

                with st.spinner("Loading resume and generating initial summary..."):
                    text = extract_text_from_pdf(pdf)
                    if not text:
                        st.error("No text found inside PDF.")
                        return

                    st.session_state.resume_text_chat = text
                    st.session_state.chat_model = model_input or DEFAULT_GEMINI_MODEL

                    # Generate initial summary as first assistant message
                    summary_task = (
                        "Summarize the resume: include 6 bullets for skills/achievements, "
                        "1-line career headline, and 2-sentence conclusion."
                    )
                    prompt = prepare_simple_prompt(summary_task, text)
                    try:
                        summary = safe_call_gemini(prompt, api_key, model=st.session_state.chat_model)
                    except Exception as e:
                        st.error(f"Gemini call failed while creating summary: {e}")
                        return

                    st.session_state.chat_history = [
                        {
                            "role": "assistant",
                            "content": "Here is a summary of your resume:",
                        },
                        {
                            "role": "assistant",
                            "content": summary,
                        },
                    ]

                streamlit_rerun()
            return  # stop here until resume is loaded

        # If resume is already loaded: show chat interface
        model_name = st.session_state.get("chat_model", DEFAULT_GEMINI_MODEL)

        # Optional reset button
        cols = st.columns([0.8, 0.2])
        with cols[1]:
            if st.button("Reset chat"):
                st.session_state.resume_text_chat = None
                st.session_state.chat_history = []
                st.session_state.chat_model = DEFAULT_GEMINI_MODEL
                streamlit_rerun()

        # Show history
        for msg in st.session_state.chat_history:
            with st.chat_message("assistant" if msg["role"] == "assistant" else "user"):
                st.markdown(msg["content"])

        # Chat input
        user_msg = st.chat_input("Ask a question about your resume...")
        if user_msg:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_msg})

            # Build a single prompt including history + resume
            history_txt = ""
            for m in st.session_state.chat_history:
                speaker = "User" if m["role"] == "user" else "Assistant"
                history_txt += f"{speaker}: {m['content']}\n"

            prompt = (
                "You are a helpful assistant answering questions about this resume.\n\n"
                f"Resume:\n{st.session_state.resume_text_chat}\n\n"
                "Conversation so far:\n"
                f"{history_txt}\n\n"
                "Answer the last user question in detail, using only the information in the resume. "
                "If something is not in the resume, say that it is not mentioned."
            )

            # Call Gemini and display answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = safe_call_gemini(prompt, api_key, model=model_name)
                    except Exception as e:
                        answer = f"Gemini call failed: {e}"
                st.markdown(answer)

            # Save assistant answer
            st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ----------------------- LinkedIn Scraper (kept logic similar to original) -----------------------
class LinkedInScraper:
    @staticmethod
    def webdriver_setup(headless=True):
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")

        try:
            driver = webdriver.Chrome(options=options)
        except Exception:
            driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
        driver.maximize_window()
        return driver

    @staticmethod
    def get_userinput():
        add_vertical_space(2)
        with st.form(key="linkedin_form"):
            add_vertical_space(1)
            col1, col2, col3 = st.columns([0.5, 0.3, 0.2], gap="medium")
            with col1:
                job_title_input = st.text_input("Job Title (comma separated)")
                job_title_input = job_title_input.split(",") if job_title_input.strip() else []
            with col2:
                job_location = st.text_input("Job Location", value="India")
            with col3:
                job_count = st.number_input("Job Count", min_value=1, value=1, step=1)
            add_vertical_space(1)
            submit = st.form_submit_button("Search LinkedIn Jobs")
            add_vertical_space(1)
        return job_title_input, job_location, job_count, submit

    @staticmethod
    def build_url(job_title, job_location):
        b = []
        for i in job_title:
            x = i.split()
            y = "%20".join(x)
            b.append(y)
        job_title = "%2C%20".join(b)
        link = f"https://in.linkedin.com/jobs/search?keywords={job_title}&location={job_location}&locationId=&geoId=102713980&f_TPR=r604800&position=1&pageNum=0"
        return link

    @staticmethod
    def open_link(driver, link):
        while True:
            try:
                driver.get(link)
                driver.implicitly_wait(5)
                time.sleep(3)
                driver.find_element(by=By.CSS_SELECTOR, value='span.switcher-tabs__placeholder-text.m-auto')
                return
            except NoSuchElementException:
                continue

    @staticmethod
    def link_open_scrolldown(driver, link, job_count):
        LinkedInScraper.open_link(driver, link)
        for i in range(0, job_count):
            body = driver.find_element(by=By.TAG_NAME, value="body")
            body.send_keys(Keys.PAGE_UP)
            try:
                driver.find_element(
                    by=By.CSS_SELECTOR,
                    value="button[data-tracking-control-name='public_jobs_contextual-sign-in-modal_modal_dismiss']>icon>svg",
                ).click()
            except:
                pass
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            driver.implicitly_wait(2)
            try:
                driver.find_element(by=By.CSS_SELECTOR, value="button[aria-label='See more jobs']").click()
                driver.implicitly_wait(5)
            except:
                pass

    @staticmethod
    def job_title_filter(scrap_job_title, user_job_title_input):
        user_input = [i.lower().strip() for i in user_job_title_input]
        scrap_title = [i.lower().strip() for i in [scrap_job_title]]
        confirmation_count = 0
        for i in user_input:
            if all(j in scrap_title[0] for j in i.split()):
                confirmation_count += 1
        if confirmation_count > 0:
            return scrap_job_title
        else:
            return np.nan

    @staticmethod
    def scrap_company_data(driver, job_title_input, job_location):
        company = driver.find_elements(by=By.CSS_SELECTOR, value='h4[class="base-search-card__subtitle"]')
        company_name = [i.text for i in company]

        location = driver.find_elements(by=By.CSS_SELECTOR, value='span[class="job-search-card__location"]')
        company_location = [i.text for i in location]

        title = driver.find_elements(by=By.CSS_SELECTOR, value='h3[class="base-search-card__title"]')
        job_title = [i.text for i in title]

        url = driver.find_elements(by=By.XPATH, value='//a[contains(@href, "/jobs/")]')
        website_url = [i.get_attribute("href") for i in url]

        df = pd.DataFrame(company_name, columns=["Company Name"])
        df["Job Title"] = pd.DataFrame(job_title)
        df["Location"] = pd.DataFrame(company_location)
        df["Website URL"] = pd.DataFrame(website_url)

        df["Job Title"] = df["Job Title"].apply(lambda x: LinkedInScraper.job_title_filter(x, job_title_input))
        df["Location"] = df["Location"].apply(lambda x: x if job_location.lower() in x.lower() else np.nan)
        df = df.dropna()
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def scrap_job_description(driver, df, job_count):
        website_url = df["Website URL"].tolist()
        job_description = []
        description_count = 0
        for i in range(0, len(website_url)):
            try:
                LinkedInScraper.open_link(driver, website_url[i])
                driver.find_element(
                    by=By.CSS_SELECTOR, value='button[data-tracking-control-name="public_jobs_show-more-html-btn"]'
                ).click()
                driver.implicitly_wait(5)
                time.sleep(1)
                description = driver.find_elements(
                    by=By.CSS_SELECTOR, value='div[class="show-more-less-html__markup relative overflow-hidden"]'
                )
                data = [j.text for j in description][0]
                if len(data.strip()) > 0 and data not in job_description:
                    job_description.append(data)
                    description_count += 1
                else:
                    job_description.append("Description Not Available")
            except:
                job_description.append("Description Not Available")
            if description_count == job_count:
                break
        df = df.iloc[: len(job_description), :]
        df["Job Description"] = pd.DataFrame(job_description, columns=["Description"])
        df["Job Description"] = df["Job Description"].apply(lambda x: np.nan if x == "Description Not Available" else x)
        df = df.dropna()
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def display_data_userinterface(df_final):
        add_vertical_space(1)
        if len(df_final) > 0:
            for i in range(0, len(df_final)):
                st.markdown(f'<h3 style="color: orange;">Job Posting Details : {i+1}</h3>', unsafe_allow_html=True)
                st.write(f"Company Name : {df_final.iloc[i,0]}")
                st.write(f"Job Title    : {df_final.iloc[i,1]}")
                st.write(f"Location     : {df_final.iloc[i,2]}")
                st.write(f"Website URL  : {df_final.iloc[i,3]}")
                with st.expander(label="Job Desription"):
                    st.write(df_final.iloc[i,4])
                add_vertical_space(3)
        else:
            st.markdown(f'<h5 style="text-align: center;color: orange;">No Matching Jobs Found</h5>', unsafe_allow_html=True)

    @staticmethod
    def main():
        driver = None
        try:
            job_title_input, job_location, job_count, submit = LinkedInScraper.get_userinput()
            add_vertical_space(2)
            if submit:
                if job_title_input != [] and job_location != "":
                    with st.spinner("Chrome Webdriver Setup Initializing..."):
                        driver = LinkedInScraper.webdriver_setup()
                    with st.spinner("Loading More Job Listings..."):
                        link = LinkedInScraper.build_url(job_title_input, job_location)
                        LinkedInScraper.link_open_scrolldown(driver, link, job_count)
                    with st.spinner("scraping Job Details..."):
                        df = LinkedInScraper.scrap_company_data(driver, job_title_input, job_location)
                        df_final = LinkedInScraper.scrap_job_description(driver, df, job_count)
                    LinkedInScraper.display_data_userinterface(df_final)
                elif job_title_input == []:
                    st.markdown(f'<h5 style="text-align: center;color: orange;">Job Title is Empty</h5>', unsafe_allow_html=True)
                elif job_location == "":
                    st.markdown(f'<h5 style="text-align: center;color: orange;">Job Location is Empty</h5>', unsafe_allow_html=True)
        except Exception as e:
            add_vertical_space(2)
            st.markdown(f'<h5 style="text-align: center;color: orange;">{e}</h5>', unsafe_allow_html=True)
        finally:
            if driver:
                driver.quit()


# ----------------------- Streamlit App -----------------------
def main_app():
    st.set_page_config(page_title="Resume Analyzer", layout="wide")
    st.title("Resume Analyzer AI")
    add_vertical_space(1)

    with st.sidebar:
        add_vertical_space(1)
        st.markdown("### Gemini / Google API")
        gemini_api_key_input = st.text_input("Gemini API Key ", type="password")
        st.caption("Provide API key here to use in the resume tools, or set GEMINI_API_KEY env var.")
        add_vertical_space(1)
        option = option_menu(
            menu_title="",
            options=["Summary", "Strength", "Weakness", "Job Titles", "ATS Score", "Chat", "Linkedin Jobs"],
            icons=["file-earmark-text", "award", "exclamation", "list", "speedometer2", "chat-dots", "linkedin"],
        )

    gemini_api_key = gemini_api_key_input.strip() or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if option == "Summary":
        ResumeAnalyzer.resume_summary_ui(gemini_api_key)
    elif option == "Strength":
        ResumeAnalyzer.resume_strength_ui(gemini_api_key)
    elif option == "Weakness":
        ResumeAnalyzer.resume_weakness_ui(gemini_api_key)
    elif option == "Job Titles":
        ResumeAnalyzer.job_title_suggestion_ui(gemini_api_key)
    elif option == "ATS Score":
        ResumeAnalyzer.ats_score_ui(gemini_api_key)
    elif option == "Chat":
        ResumeChat.chat_ui(gemini_api_key)
    elif option == "Linkedin Jobs":
        LinkedInScraper.main()


if __name__ == "__main__":
    main_app()
