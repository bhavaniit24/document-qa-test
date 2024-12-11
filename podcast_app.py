import os, re, datetime, glob
from operator import itemgetter
from PyPDF2 import PdfReader
from pydub import AudioSegment
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI


class PodcastApp:
    plan_prompt = ChatPromptTemplate.from_template(
        """You are a very clever planner of podcast scripts. You will be given the text of a document, and your task will be to generate a plan for a podcast involving 3 persons discussing the content of the paper in a very engaging, interactive and enthusiastic way. The plan will be structured using titles and bullet points only. The plan for the podcast should follow the structure of the paper.

        The podcast involves the following persons:
        - The host: presents the paper in a professional, friendly, warm, and enthusiastic manner.
        - The learner: asks clever and significant questions, is curious and funny.
        - The expert: provides deep insights, comments, and details, speaks less but with more profound detail.

        Example structure:
        # Title: title of the podcast
        # Section 1: title of section 1
        - bullet point 1
        - bullet point 2
        ...
        # Section 2: title of section 2
        ...
        # Section n: title of section n

        The paper: {paper}
        The podcast plan in titles and bullet points:"""
    )

    discuss_prompt_template = ChatPromptTemplate.from_template(
        """You are a very clever scriptwriter of podcast discussions. You will be given a plan for a section of a podcast (involving 3 persons) that started already. Your task is to generate a brief dialogue for the given section, without introductions, focusing on making it engaging, interactive, enthusiastic, and clever. Follow the structure of the plan.

        Podcast roles:
        - Host: engaging, professional, friendly, warm, enthusiastic.
        - Learner: asks clever and significant questions, curious and funny.
        - Expert: provides deep, insightful comments, profound and less frequent interventions.

        Section plan: {section_plan}
        Previous dialogue (to avoid repetitions): {previous_dialogue}
        Additional context:{additional_context}
        Brief section dialogue:"""
    )

    initial_dialogue_prompt = ChatPromptTemplate.from_template(
        """You are a very clever scriptwriter of podcast introductions. Given the title and a brief glimpse of a document, generate a 3-interaction introduction. Do not use sound effects. The introduction should be captivating, interactive, and make listeners eager to hear more. Do not end with the host; end with the expert.

        Podcast roles:
        - Host: presents the paper in a professional, friendly, warm, enthusiastic manner.
        - Learner: asks clever and significant questions, curious and funny.
        - Expert: provides deep insights, related topics.

        Content of the paper: {paper_head}
        Brief 3 interactions introduction:"""
    )

    enhance_prompt = ChatPromptTemplate.from_template(
        """You are a very clever scriptwriter of podcast discussions. Given a draft script, enhance it by removing audio effect mentions, reducing repetition, and improving transitions and twists. Do not mention any audio effects or how people are speaking, only their dialogues. Just refine the text.

        The draft script:
        {draft_script}
        The enhanced script:"""
    )

    def __init__(self, api_key=None, llm_model="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.llm = ChatOpenAI(model=llm_model, api_key=api_key)
        self.chains = {
            "plan_script_chain": self.plan_prompt | self.llm | self.parse_plan,
            "initial_dialogue_chain": self.initial_dialogue_prompt
            | self.llm
            | StrOutputParser(),
            "enhance_chain": self.enhance_prompt | self.llm | StrOutputParser(),
        }

    def tts(self, speaker, text, outdir):
        voice_map = {"Host": "alloy", "Learner": "nova", "Expert": "fable"}
        now = int(datetime.datetime.now().timestamp())
        r = self.client.audio.speech.create(
            model="tts-1", voice=voice_map.get(speaker), input=text
        )
        r.stream_to_file(f"./{outdir}/{speaker.lower()}_{now}.mp3")

    def ts_from_fname(self, f):
        m = re.search(r"(\d{10})", f)
        return int(m.group(0)) if m else 0

    def merge_mp3(self, d, out):
        fs = sorted(
            [os.path.basename(x) for x in glob.glob(f"./{d}/*.mp3")],
            key=self.ts_from_fname,
        )
        audio = AudioSegment.empty()
        for f in fs:
            audio += AudioSegment.from_mp3(f"./{d}/{f}")
        audio.export(out, format="mp3")

    def podcast_audio(self, script):
        od = f"podcast_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        os.mkdir(od)
        lines = re.findall(
            r"(Host|Learner|Expert):\s*(.*?)(?=(Host|Learner|Expert|$))",
            script,
            re.DOTALL,
        )
        for sp, txt, _ in lines:
            txt = txt.strip()
            if txt:
                self.tts(sp, txt, od)
        now = int(datetime.datetime.now().timestamp())
        self.merge_mp3(od, f"podcast_{now}.mp3")

    def read_pdf(self, pdf):
        reader = PdfReader(pdf)
        text, head, c = [], [], True
        for page in reader.pages:
            t = page.extract_text()
            if not t:
                continue
            if c:
                text.append(t)
                if "Conclusion" in t:
                    c = False
            if "Introduction" in t and not head:
                idx = t.index("Introduction")
                head.append(t[:idx])
        return "\n".join(text), "\n".join(head)

    def parse_plan(self, ai_msg: AIMessage):
        lines = ai_msg.content.strip().splitlines()[1:]
        sections, cur = [], []
        for l in lines:
            if re.match(r"^#+\s", l):
                if cur:
                    sections.append(" ".join(cur))
                    cur = []
                cur.append(l.strip())
            elif re.match(r"^- ", l):
                cur.append(l.strip())
        if cur:
            sections.append(" ".join(cur))
        return sections

    def chain_discuss(self, txt_file):
        loader = TextLoader(txt_file, encoding="UTF-8")
        docs = loader.load()
        s = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(docs)
        vs = Chroma.from_documents(
            documents=s, embedding=OpenAIEmbeddings()
        ).as_retriever()
        fmt = lambda d: "\n\n".join(x.page_content for x in d)
        return (
            {
                "additional_context": itemgetter("section_plan") | vs | fmt,
                "section_plan": itemgetter("section_plan"),
                "previous_dialogue": itemgetter("previous_dialogue"),
            }
            | self.discuss_prompt_template
            | self.llm
            | StrOutputParser()
        )

    def gen_script(self, pdf):
        start = datetime.datetime.now()
        txt_file = f"text_paper_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
        full_text, head = self.read_pdf(pdf)
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(full_text)

        plan = self.chains["plan_script_chain"].invoke({"paper": full_text})
        init = self.chains["initial_dialogue_chain"].invoke({"paper_head": head})

        rag = self.chain_discuss(txt_file)
        script, actual = init, init
        for sec in plan:
            sc = rag.invoke({"section_plan": sec, "previous_dialogue": actual})
            script += sc
            actual = sc

        final = self.chains["enhance_chain"].invoke({"draft_script": script})
        print(f"Time taken: {datetime.datetime.now()-start}")
        return final

    def podcastify(self, pdf):
        print("Generating script...")
        sc = self.gen_script(pdf)
        print("Script done. Generating audio...")
        self.podcast_audio(sc)
        print("Podcast done!")
