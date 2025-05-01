# App
import re
import sys
import settings
import traceback
import gradio as gr

from colorama import Fore
from typing import List, Tuple
from utils.models import get_model
from utils.prompt import process_prompt
from utils.files import process_files, load_json_to_dataframe

# Avatars
user_avatar = "./assets/images/user_avatar.jpg"
bot_avatar = "./assets/images/bot_avatar.gif"

# Read CSS
with open("./assets/styles.css", "r") as f:
    css = f.read()


with gr.Blocks(
    title="ChatRAG",
    css=css,
    fill_width=True,
) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="group-1"):
                with gr.Row():
                    rag_mode = gr.Radio(
                        choices=["ON", "OFF"],
                        scale=1,
                        label=f"RAG mode - {settings.VECSTORAGE.capitalize()} storage",
                        value="ON",
                        elem_classes="radio-1",
                    )
                with gr.Accordion(label="RAG Settings", open=False):
                    n_ctx_chunks = gr.Number(
                        label="Context chunks",
                        minimum=0,
                        maximum=10,
                        step=1,
                        value=5,
                        elem_classes="number-1",
                    )
                    with gr.Row():
                        txt_splitter = gr.Radio(
                            label="Text splitter",
                            choices=[
                                "CharacterTextSplitter",
                                "RecursiveCharacterTextSplitter",
                                "SemanticChunker",
                            ],
                            scale=1,
                            value="RecursiveCharacterTextSplitter",
                            elem_classes="radio-2",
                        )
                    with gr.Row():
                        chunks_size = gr.Number(
                            label="Chunks size",
                            minimum=100,
                            maximum=3000,
                            step=100,
                            value=1000,
                            elem_classes="number-1",
                        )
                        chunks_overlap = gr.Number(
                            label="Chunks overlap",
                            minimum=10,
                            maximum=500,
                            step=10,
                            value=200,
                            elem_classes="number-1",
                        )
                    with gr.Row():
                        pdf_loader = gr.Radio(
                            label="PDF loader",
                            choices=[
                                "PyPDFLoader",
                                "PDFPlumberLoader",
                            ],
                            scale=1,
                            value="PyPDFLoader",
                            elem_classes="radio-2",
                        )

                    with gr.Row():
                        csv_loader = gr.Radio(
                            label="CSV loader",
                            choices=[
                                "CSVLoader",
                                "UnstructuredCSVLoader",
                            ],
                            scale=1,
                            value="CSVLoader",
                            elem_classes="radio-2",
                        )

            with gr.Group(elem_classes="group-1"):
                model_dropdown = gr.Dropdown(
                    choices=settings.MODELS,
                    value=settings.MODELS[0],
                    label="Set model",
                    show_label=True,
                    interactive=True,
                    elem_classes="dropdown-1",
                )

                with gr.Accordion(label="LLM Settings", open=False):
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.0,
                        step=0.1,
                        label="Temperature",
                        interactive=True,
                        elem_classes="slider-1"
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.1,
                        label="Top_p",
                        interactive=True,
                    )

            with gr.Group(elem_classes="group-1"):
                file_upload = gr.File(
                    file_types=[".pdf", ".csv", ".xls", ".xlsx", ".docx"],
                    file_count="multiple",
                    label="Upload de arquivos",
                    interactive=True,
                    height=160,
                )
                status_area = gr.Textbox(show_label=False, elem_classes="textbox-2")
                run_dbvec = gr.Button(
                    value=f"Load to {settings.VECSTORAGE.capitalize()}", variant="primary", elem_classes="button-1"
                )
                run_dbvec.click(
                    fn=process_files,
                    inputs=[
                        file_upload,
                        txt_splitter,
                        pdf_loader,
                        chunks_size,
                        chunks_overlap,
                        csv_loader,
                    ],
                    outputs=status_area,
                )

            with gr.Group(elem_classes="group-1"):
                update_data = gr.Button(
                    value="Refresh uploaded files",
                    variant="huggingface",
                    elem_classes="button-2",
                )
                file_dataframe = gr.DataFrame(
                    headers=["File", "Size", "Type"],
                    wrap=None,
                    elem_classes="dataframe-container",
                    value=load_json_to_dataframe,
                    max_height=185,
                    interactive=False,
                )

                update_data.click(
                    fn=load_json_to_dataframe, inputs=None, outputs=file_dataframe
                )

        with gr.Column(scale=3):
            with gr.Group(elem_classes="group-1"):
                chatbot = gr.Chatbot(
                    elem_classes="chatbot",
                    type="messages",
                    min_height=790,
                    avatar_images=(user_avatar, bot_avatar),
                    layout="bubble",
                    label="RAG-CHATBOT"
                )

                msg = gr.Textbox(
                    placeholder="Ask something...",
                    show_label=False,
                    elem_classes="textbox-1",
                )

    def get_response(
        user_msg: str,
        chat_history: List[dict],
        selected_model: str,
        temperature: float,
        top_p: float,
        rag_mode: str,
        n_ctx_chunks: int,
    ) -> Tuple[str, List[dict]]:
        
        """
        Generates a chatbot response based on the user's message and conversation history.

        This function uses a selected language model to generate a response, processes the user's input
        and returns the response generated by the model. The conversation history is maintained and updated with each new interaction.

        Parameters:
            user_msg (str): The user's message that will be processed by the chatbot.
            chat_history (List[dict]): The conversation history containing previous messages between the user and the assistant.
            selected_model (str): The name of the language model that will be used to generate the response.
            temperature (float): The temperature parameter to control the creativity of the response (the higher the value, the more creative).
            top_p (float): The parameter to control the diversity of the response, using the per-core sampling method.
            rag_mode (str): Chat with RAG enabled "ON" or disabled "OFF".
            n_ctx_chunks (int): The number of context chunks to consider when generating the response, used in RAG.

        Returns:
            Return (Tuple[str, List[dict]]): Returns a tuple where the first value is an empty string (for compatibility) and
            the second value is the conversation history updated with the assistant's response. 
        """
        
        llm = get_model(selected_model, temperature, top_p)
        chat_history.append({"role": "user", "content": user_msg})
        messages = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in chat_history]
        )
        try:
            prompt = process_prompt(messages, user_msg, rag_mode, n_ctx_chunks)
            bot_message = llm.invoke(prompt).content
            bot_message = re.sub(
                r"<think>.*?</think>|<think>|</think>", "", bot_message, flags=re.DOTALL
            )

        except Exception as e:
            exception_name = type(e).__name__
            track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
            bot_message = f"EXCEPTION ERROR: {exception_name}: {track_line}"
            print(f"\n{Fore.LIGHTRED_EX}{bot_message}{Fore.RESET}")
            #raise sys.exc_info()[0]

        chat_history.append({"role": "assistant", "content": bot_message})

        return "", chat_history

    msg.submit(
        get_response,
        [
            msg,
            chatbot,
            model_dropdown,
            temperature_slider,
            top_p_slider,
            rag_mode,
            n_ctx_chunks,
        ],
        [msg, chatbot],
    )


if __name__ == "__main__":
    demo.launch(
        auth=settings.USERS,
        favicon_path=bot_avatar,
        server_name="0.0.0.0",
        server_port=7860,
    )
